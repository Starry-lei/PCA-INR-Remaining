import torch
from torch import nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_regular
from .pde_layer import PDELayer
from .shared_definition import NONLINEARITIES
import numpy as np
import torch.nn.functional as F

class ImNet(nn.Module):
    """ImNet layer pytorch implementation."""

    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=4,
        nf=32,
        nonlinearity="leakyrelu",
    ):
        """Initialization.

        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          nf: int, width of the second to last layer.
          activation: tf activation op.
          name: str, name of the layer.
        """
        super(ImNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.activ = NONLINEARITIES[nonlinearity]
        self.fc0 = nn.Linear(self.dimz, nf * 16)
        self.fc1 = nn.Linear(nf * 16 + self.dimz, nf * 8)
        self.fc2 = nn.Linear(nf * 8 + self.dimz, nf * 4)
        self.fc3 = nn.Linear(nf * 4 + self.dimz, nf * 2)
        self.fc4 = nn.Linear(nf * 2 + self.dimz, nf * 1)
        self.fc5 = nn.Linear(nf * 1, out_features)
        self.fc = [self.fc0, self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        """Forward method.

        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
        x_tmp = x
        for dense in self.fc[:4]:
            x_tmp = self.activ(dense(x_tmp))
            x_tmp = torch.cat([x_tmp, x], dim=-1)
        x_tmp = self.activ(self.fc4(x_tmp))
        x_tmp = self.fc5(x_tmp)
        return x_tmp


class VanillaNet(nn.Module):
    """Vanilla mpl pytorch implementation."""

    def __init__(
        self,
        dim=3,
        in_features=32,
        out_features=3,
        nf=50,
        nlayers=4,
        nonlinearity="leakyrelu",
    ):
        """Initialization.

        Args:
          dim: int, dimension of input points.
          in_features: int, length of input features (i.e., latent code).
          out_features: number of output features.
          nf: int, width of the second to last layer.
          nlayers: int, number of layers in mlp (inc. input/output layers).
          activation: tf activation op.
          name: str, name of the layer.
        """
        super(VanillaNet, self).__init__()
        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features
        self.out_features = out_features
        self.nf = nf
        self.nlayers = nlayers
        self.activ = NONLINEARITIES[nonlinearity]
        modules = [nn.Linear(dim + in_features, nf), self.activ]
        assert nlayers >= 2
        for i in range(nlayers - 2):
            modules += [nn.Linear(nf, nf), self.activ]
        modules += [nn.Linear(nf, out_features)]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        """Forward method.

        Args:
          x: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
        return self.net(x)


def symmetrize(net, latent_vector, points, symm_dim):
    """Make network output symmetric."""
    # query both sides of the symmetric dimension
    points_pos = points
    points_neg = points.clone()
    points_neg[..., symm_dim] = -points_neg[..., symm_dim]
    y_pos = net(latent_vector, points_pos)
    y_neg = net(latent_vector, points_neg)
    y_sym = (y_pos + y_neg) / 2
    y_sym[..., symm_dim] = (y_pos[..., symm_dim] - y_neg[..., symm_dim]) / 2
    return y_sym


class DeformationFlowNetwork(nn.Module):
    def __init__(
        self,
        dim=3,
        latent_size=1,
        nlayers=4,
        width=50,
        nonlinearity="leakyrelu",
        arch="imnet",
        divfree=False,
    ):
        """Intialize deformation flow network.

        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
          divfree: bool, paramaterize a divergence free flow.
        """
        super(DeformationFlowNetwork, self).__init__()
        self.dim = dim
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        self.nonlinearity = nonlinearity

        self.arch = arch
        self.divfree = divfree

        assert arch in ["imnet", "vanilla"]
        if arch == "imnet":
            self.net = ImNet(
                dim=dim,
                in_features=latent_size,
                out_features=dim,
                nf=width,
                nonlinearity=nonlinearity,
            )
        else:  # vanilla
            self.net = VanillaNet(
                dim=dim,
                in_features=latent_size,
                out_features=dim,
                nf=width,
                nlayers=nlayers,
                nonlinearity=nonlinearity,
            )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                # nn.init.normal_(m.weight, mean=0, std=1)
                nn.init.constant_(m.bias, val=0)
        if divfree:
            self.curl = self._get_curl_layer()

    def _get_curl_layer(self):
        in_vars = "x, y, z"
        out_vars = "u, v, w"
        eqn_strs = [
            "dif(w, y) - dif(v, z)",
            "dif(u, z) - dif(w, x)",
            "dif(v, x) - dif(u, y)",
        ]
        eqn_names = [
            "curl_x",
            "curl_y",
            "curl_z",
        ]  # a name/identifier for the equations
        curl = PDELayer(
            in_vars=in_vars, out_vars=out_vars
        )  # initialize the pde layer
        for eqn_str, eqn_name in zip(eqn_strs, eqn_names):  # add equations
            curl.add_equation(eqn_str, eqn_name)
        return curl

    def forward(self, latent_vector, points):
        """
        Args:
          latent_vector: tensor of shape [batch, latent_size], latent code for
            each shape
          points: tensor of shape [batch, num_points, dim], points representing
            each shape

        Returns:
          velocities: tensor of shape [batch, num_points, dim], velocity at
            each point
        """
        latent_vector = latent_vector.unsqueeze(1).expand(
            -1, points.shape[1], -1
        )  # [batch, num_points, latent_size]
        # wrapper for pde layer

        def fwd_fn(points):
            """Forward function.

            Where inpt[..., 0], inpt[..., 1], inpt[..., 2] correspond to x, y,
              z and
            out[..., 0], out[..., 1], out[..., 2] correspond to u, v, w
            """
            points_latents = torch.cat(
                (points, latent_vector), axis=-1
            )  # [batch, num_points, dim + latent_size]
            b, n, d = points_latents.shape
            res = self.net(points_latents.reshape([-1, d]))
            res = res.reshape([b, n, self.dim])
            return res

        if self.divfree:
            # return the curl of the velocity field instead
            self.curl.update_forward_method(fwd_fn)
            _, res_dict = self.curl(points)  # res are the equation residues
            res = torch.cat(
                [res_dict["curl_x"], res_dict["curl_y"], res_dict["curl_z"]],
                dim=-1,
            )  # [batch, num_points, dim]
            return res
        else:
            return fwd_fn(points)


class ConformalDeformationFlowNetwork(nn.Module):
    def __init__(
        self,
        dim=3,
        latent_size=1,
        nlayers=4,
        width=50,
        nonlinearity="softplus",
        output_scalar=False,
        arch="imnet",
    ):
        """Intialize conformal deformation flow network w/ irrotational flow.

        The network produces a scalar field Phi(x,y,z,t), and the velocity
        field is represented as the gradient of Phi.
        v = \nabla\Phi  # noqa: W605
        The gradients can be efficiently computed as the Jacobian through
        backprop.

        Args:
          dim: int, physical dimensions. Either 2 for 2d or 3 for 3d.
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
        """
        super(ConformalDeformationFlowNetwork, self).__init__()
        self.dim = dim
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width
        self.nonlinearity = nonlinearity
        self.output_scalar = output_scalar

        self.scale = nn.Parameter(torch.ones(1) * 0.01)

        nlin = NONLINEARITIES[nonlinearity]

        modules = [nn.Linear(dim + latent_size, width), nlin]
        for i in range(nlayers - 2):
            modules += [nn.Linear(width, width), nlin]
        modules += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, latent_vector, points):
        """
        Args:
          latent_vector: tensor of shape [batch, latent_size], latent code for
            each shape
          points: tensor of shape [batch, num_points, dim], points representing
            each shape
        Returns:
          velocities: tensor of shape [batch, num_points, dim], velocity at
            each point
        """
        latent_vector = latent_vector.unsqueeze(1).expand(
            -1, points.shape[1], -1
        )  # [batch, num_points, latent_size]
        b, num_points, lat_size = latent_vector.shape
        latent_flat = latent_vector.reshape([-1, lat_size])
        points_flat = points.reshape([-1, self.dim])
        points_flat_ = torch.autograd.Variable(points_flat)
        points_flat_.requires_grad = True
        points_latents = torch.cat(
            (points_flat_, latent_flat), axis=-1
        )  # [batch*num_points, dim + latent_size]
        phi = self.net(points_latents) * self.scale
        vel_flat = torch.autograd.grad(
            phi,
            points_flat_,
            grad_outputs=torch.ones_like(phi),
            create_graph=True,
        )[0]
        vel = vel_flat.reshape(points.shape)
        if self.output_scalar:
            return vel, phi
        else:
            return vel


class DeformationSignNetwork(nn.Module):
    def __init__(
        self, latent_size=1, nlayers=3, width=20, nonlinearity="tanh"
    ):
        """Initialize deformation sign network.
        Args:
          latent_size: int, size of latent space. >= 1.
          nlayers: int, number of neural network layers. >= 2.
          width: int, number of neurons per hidden layer. >= 1.
        """
        super(DeformationSignNetwork, self).__init__()
        self.latent_size = latent_size
        self.nlayers = nlayers
        self.width = width

        nlin = NONLINEARITIES[nonlinearity]
        modules = [nn.Linear(latent_size, width, bias=False), nlin]
        for i in range(nlayers - 2):
            modules += [nn.Linear(width, width, bias=False), nlin]
        modules += [nn.Linear(width, 1, bias=False), nlin]
        self.net = nn.Sequential(*modules)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1e-1)

    def forward(self, dir_vector):
        """
        Args:
          dir_vector: tensor of shape [batch, latent_size], latent direction.
        Returns:
          signs: tensor of shape [batch, 1, 1]
        """
        dir_vector = dir_vector / (
            torch.norm(dir_vector, dim=-1, keepdim=True) + 1e-6
        )  # normalize
        signs = self.net(dir_vector).unsqueeze(-1)
        return signs


class NeuralFlowModel(nn.Module):
    def __init__(
        self,
        dim=3,
        latent_size=1,
        f_nlayers=4,
        f_width=50,
        s_nlayers=3,
        s_width=20,
        nonlinearity="relu",
        conformal=False,
        arch="imnet",
        no_sign_net=False,
        divfree=False,
        symm_dim=None,
    ):
        super(NeuralFlowModel, self).__init__()
        if conformal:
            model = ConformalDeformationFlowNetwork

        else:
            model = DeformationFlowNetwork
        self.no_sign_net = no_sign_net
        self.flow_net = model(
            dim=dim,
            latent_size=latent_size,
            nlayers=f_nlayers,
            width=f_width,
            nonlinearity=nonlinearity,
            arch=arch,
            divfree=divfree,
        )
        if not no_sign_net:
            self.sign_net = DeformationSignNetwork(
                latent_size=latent_size, nlayers=s_nlayers, width=s_width
            )
        self.symm_dim = symm_dim
        self.latent_source = None
        self.latent_target = None
        self.latent_updated = False
        self.conformal = conformal
        self.arch = arch
        self.encoder = None
        self.lat_params = None
        self.scale = nn.Parameter(torch.ones(1) * 1e-1)
        self.latent_sequence=None
        # self.scale = nn.Parameter(torch.ones(1))
        

    def add_encoder(self, encoder):
        self.encoder = encoder

    def add_lat_params(self, lat_params):
        self.lat_params = lat_params

    def get_lat_params(self, idx):
        assert self.lat_params is not None
        return self.lat_params[idx]

    def update_latents(self, latent_sequence):
        """
        Args:
            latent_sequence: long or float tensor of shape
                [batch, nsteps, latent_size].
                sequence of latents along deformation path.
                if long, index into self.lat_params to retrieve latents.
        Returns:
            latent_waypoint: float tensor of shape [batch, nsteps], interp
                coefficient betwene [0, 1] corresponding to each latent code.
        """
        bs, ns, d = latent_sequence.shape
        dev = latent_sequence.device
        self.latent_sequence = latent_sequence
        self.latent_seq_len = torch.norm(
            self.latent_sequence[:, 1:] - self.latent_sequence[:, :-1], dim=-1
        )  # [batch, nsteps-1]
        self.latent_seq_len_sum = torch.sum(
            self.latent_seq_len, dim=1
        )  # [batch]
        self.latent_seq_weight = (
            self.latent_seq_len / self.latent_seq_len_sum[:, None]
        )  # [batch, nsteps-1]


        # print("show self.latent_seq_weight!!!:",self.latent_seq_weight)#[1,1]
        # show self.latent_seq_weight!!!: tensor([[1.],[1.]], device='cuda:0')

    

        self.latent_seq_bins = torch.cumsum(
            self.latent_seq_weight, dim=1
        )  # [batch, nsteps-1]

        # print("show self.self.latent_seq_bins!!!:",self.latent_seq_bins)#[1,1]


        self.latent_seq_bins = torch.cat(
            [torch.zeros([bs, 1], device=dev), self.latent_seq_bins], dim=1
        )  # [batch, nsteps]


        # print("show self.self.self.latent_seq_bins!!!:",self.latent_seq_bins)#[1,1]


        self.latent_updated = True

        return self.latent_seq_bins

    def twopoints_latent_at_t(self, t, return_sign=False):
        """Helper fn to compute latent at t."""


        t = t.to(self.latent_seq_bins.device)
        # t = t.detach()
        # find out which bin this t falls into
        # Linear interpolation coefficient
        alpha = (t - self.latent_seq_bins[:, 0]) / (self.latent_seq_bins[:, 1] - self.latent_seq_bins[:, 0])  # [batch]
        
        # alpha=alpha.requires_grad_(False)
        latent_t0 = self.latent_sequence[:, 0]  # [batch, latent_size]
        latent_t1 = self.latent_sequence[:, 1]  # [batch, latent_size]
        print("check grad of latent_t0:",latent_t0.requires_grad) # True
        print("check grad of latent_t1:",latent_t1.requires_grad) # True
        print("check grad of alpha:",alpha.requires_grad) # False
        print("check grad of self.latent_seq_bins:",self.latent_seq_bins.requires_grad) # True
        latent_val = latent_t0 + alpha.unsqueeze(-1) * (latent_t1 - latent_t0)


        print("check grad of latent_val:",latent_val.requires_grad) # False
        exit()

        # print("check grad of latent_t0:",latent_t0.requires_grad) # False
        # print("check val of latent_t0_n:",latent_t0_n) 

       

        print("shape of batch_idx:",batch_idx.shape)
        print("shape of bin_idx:",bin_idx.shape)
        print("dtype of batch_idx:",batch_idx.dtype)
        print("dtype of bin_idx:",bin_idx.dtype)

        latent_t1 = self.latent_sequence[batch_idx, bin_idx + 1]  # [batch, latent_size]
        print("check grad of latent_t1:", latent_t1.requires_grad)
        print("check val of latent_t1:", latent_t1)


 
    
        








        # check grad of latent_t1: False
        exit()


        latent_val = latent_t0 + alpha[:, None] * (latent_t1 - latent_t0)

        
        latent_dir = (latent_t1 - latent_t0) / torch.norm(
            latent_t1 - latent_t0, dim=1, keepdim=True
        )
        zeros = torch.zeros_like(latent_t0)
        outward = torch.norm(latent_t0 - zeros, dim=1) < 1e-6  # [batch]
        # So -1 indicates the latent point is not near the origin (norm > 1e-6), which affects the direction of transformation.
        sign = (outward.float() - 0.5) * 2

        return latent_val, latent_dir, sign

    def latent_at_t(self, t, return_sign=False):
        """Helper fn to compute latent at t."""
        t = t.to(self.latent_seq_bins.device)
        # find out which bin this t falls into
        bin_mask = (t > self.latent_seq_bins[:, :-1]) * (
            t < self.latent_seq_bins[:, 1:]
        )
        # logical and
        bin_mask = bin_mask.float()
        bin_idx = torch.argmax(bin_mask, dim=1)  # [batch,]
        batch_idx = torch.arange(bin_idx.shape[0]).to(bin_idx.device)

        # ends of the bin
        t0 = self.latent_seq_bins[batch_idx, bin_idx]
        t1 = self.latent_seq_bins[batch_idx, bin_idx + 1]  # [batch]
        alpha = (t - t0) / (t1 - t0)  # [batch]            
        latent_t0 = self.latent_sequence[
            batch_idx, bin_idx
        ]  # [batch, latent_size]

        

        latent_t1 = self.latent_sequence[batch_idx, bin_idx + 1]  # [batch, latent_size]
        latent_val = latent_t0 + alpha[:, None] * (latent_t1 - latent_t0)
        latent_dir = (latent_t1 - latent_t0) / torch.norm(
            latent_t1 - latent_t0, dim=1, keepdim=True
        )
        zeros = torch.zeros_like(latent_t0)
        outward = torch.norm(latent_t0 - zeros, dim=1) < 1e-6  # [batch]
        # So -1 indicates the latent point is not near the origin (norm > 1e-6), which affects the direction of transformation.
        sign = (outward.float() - 0.5) * 2

        # print("check grad of latent_val:",latent_val.requires_grad) # False
        # exit()

        return latent_val, latent_dir, sign

    def forward(self, t, points):
        """
        Args:
          t: float, deformation parameter between 0 and 1.
          points: [batch, num_points, dim]
        Returns:
          vel: [batch, num_points, dim]
        """
        
        # Reparametrize eval along latent path as a function of a single
        # scalar t
        if not self.latent_updated:
            raise RuntimeError(
                "Latent not updated. "
                "Use .update_latents() to update the source and target latents"
            )

        if self.training:

            with torch.set_grad_enabled(True):

                points.requires_grad_(True)
                latent_val, latent_dir, sign = self.latent_at_t(t)
                # print("show self.scale:",self.scale)
                sign = sign[:, None, None] * self.scale
                if self.symm_dim is None:
                    # print("symm_dim show points grad:",points.requires_grad)
                    flow = self.flow_net(latent_val, points)  # [batch, num_pints, dim]
                else:
                    flow = symmetrize(self.flow_net, latent_val, points, self.symm_dim)
                # Normalize velocity based on time space proportional to latent
                # difference.
                # print("before show flow:",flow)


                # print("see self.latent_seq_len_sum[:, None, None]:",self.latent_seq_len_sum[:, None, None])
                flow *= self.latent_seq_len_sum[:, None, None]
                # print("show flow after :",flow)
                if not self.no_sign_net:
                    sign = self.sign_net(latent_dir)
                flow_signed=flow * sign
            return flow_signed
        
        else:
            latent_val, latent_dir, sign = self.latent_at_t(t)
            sign = sign[:, None, None] * self.scale
            if self.symm_dim is None:
                # print("symm_dim show points grad:",points.requires_grad)
                flow = self.flow_net(latent_val, points)  # [batch, num_pints, dim]
            else:
                flow = symmetrize(self.flow_net, latent_val, points, self.symm_dim)
           
            flow *= self.latent_seq_len_sum[:, None, None]
            # print("show flow after :",flow)
            if not self.no_sign_net:
                sign = self.sign_net(latent_dir)
            flow_signed=flow * sign
            return flow_signed



class Model(nn.Module):
    def __init__(self, args):
        super(Model,self).__init__()

        self.method = args.solver
        self.conformal = False
        self.lat_dims= args.lat_dims
        self.arch = args.arch
        self.adjoint = args.adjoint
        self.nonlinearity= args.nonlin
        self.no_sign_net= True # try sign net 
        self.deformer_nf= args.deformer_nf


        self.odeint = odeint_adjoint if args.adjoint else odeint_regular
        self.__timing = torch.from_numpy(
            np.array([0.0, 1.0]).astype("float32")
        ).to(args.device)
        self.return_waypoints = args.return_waypoints
        self.use_latent_waypoints = args.use_latent_waypoints
        self.rtol = args.rtol
        self.atol = args.atol
        self.via_hub = args.via_hub
        self.symm_dim = (2 if args.symm else None)


        self.latent_dim=64
        self.latent_en= nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),  # Normalize scale
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.net = NeuralFlowModel(
            dim=3,
            latent_size=self.lat_dims,
            f_nlayers=4,
            f_width=self.deformer_nf,
            s_nlayers=2,
            s_width=5,
            arch=self.arch,
            conformal=self.conformal,
            nonlinearity=self.nonlinearity,
            no_sign_net=self.no_sign_net,
            divfree=False,
            symm_dim=(2 if args.symm else None),
        )
        if self.symm_dim is not None:
            if not (isinstance(self.symm_dim, int) or isinstance(self.symm_dim, list)):
                raise ValueError(
                    "symm_dim must be int or list of ints, indicating axes of"
                    "symmetry."
                )

    @property
    def adjoint(self):
        return self.__adjoint

    @adjoint.setter
    def adjoint(self, isadjoint):
        assert isinstance(isadjoint, bool)
        self.__adjoint = isadjoint
        self.odeint = odeint_adjoint if isadjoint else odeint_regular

    @property
    def timing(self):
        return self.__timing

    @timing.setter
    def timing(self, timing):
        assert isinstance(timing, torch.Tensor)
        assert timing.ndim == 1
        self.__timing = timing

    def add_encoder(self, encoder):
        self.net.add_encoder(encoder)

    def add_lat_params(self, lat_params):
        self.net.add_lat_params(lat_params)

    def get_lat_params(self, idx):
        return self.net.get_lat_params(idx)

    def forward(self, points, latent_sequence):

        """Forward transformation (source -> latent_path -> target).

        To perform backward transformation, simply switch the order of the lat
        codes.

        Args:
          points: [batch, num_points, dim]
          latent_sequence: float tensor of shape [batch, nsteps, latent_size],
            ------- or -------
            long tensor of shape [batch, nsteps]
            sequence of latents along deformation path.
            if long, index into self.lat_params to retrieve latents.
        Returns:
          points_transformed:
            tensor of shape [batch, num_points, dim] if not
            self.return_waypoint.
            tensor of shape [nsteps, batch, num_points, dim] if
            self.return_waypoint.
        """

        #latent_sequence: shape torch.Size([8, 2, 64])

        if latent_sequence.dtype == torch.long:
            latent_sequence = self.get_lat_params(
                latent_sequence
            )  # [nsteps, batch, lat_dim]
            if self.via_hub:
                assert latent_sequence.shape[1] == 2
                zeros = torch.zeros_like(latent_sequence[:, :1])
                lat0 = latent_sequence[:, 0:1]
                lat1 = latent_sequence[:, 1:2]
                latent_sequence = torch.cat(
                    [lat0, zeros, lat1], dim=1
                )  # [batch, nsteps=3, lat_dim]
        
        



        points = points.requires_grad_(True)
        latent_sequence = latent_sequence.requires_grad_(True)


        # Integrate a pre-Computed PCA Basis into the deformation flow field

        
        # print("see latent_sequence:",latent_sequence.shape)# torch.Size([2, 2, 64])
        is_nonzero = torch.any(latent_sequence != 0, dim=-1)
        encoded = torch.zeros_like(latent_sequence)
        encoded[is_nonzero] = self.latent_en(latent_sequence[is_nonzero])
        
 
        waypoints = self.net.update_latents(latent_sequence)

        # print("see waypoints shape:",waypoints.shape) #  torch.Size([8, 2])
        # print("see waypoints values:",waypoints) #  torch.Size([8, 2])
        

        if self.use_latent_waypoints:
            timing = waypoints[0]
        else:
            timing = self.timing

        timing = self.timing.clone().requires_grad_(True)
        # print("see timing vals:",timing)# [0., 1.]
        points_transformed = self.odeint(
            self.net,
            points,
            timing,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
            # adjoint_options={'requires_grad': True}
        )


        # print("show waypoints requires_grad:", waypoints.requires_grad) # True
        # print("show points_transformed requires_grad:", points_transformed.requires_grad)# True
        # print("show points requires_grad:", points.requires_grad) # True
        # print("show timing grad:", timing.requires_grad) # True


        deform_magnitude = torch.norm(points_transformed - points).item()
        print(f"Deformation magnitude: {deform_magnitude}")
        # print(f"Flow network grad norm: {torch.norm(next(self.net.flow_net.parameters()).grad).item()}"
        # exit()


        deform_abs = torch.mean(
                torch.norm(points - points_transformed, dim=-1)
            )
        print("deform_abs deformation!",deform_abs)
        if deform_abs < 1e-6:
            print("no deformation!")
            exit()

        # exit()
        if self.return_waypoints:
            return points_transformed
        else:
            return points_transformed[-1]
