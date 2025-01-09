import torch
from torch import nn
from torchdiffeq import odeint_adjoint
from torchdiffeq import odeint as odeint_regular
from .pde_layer import PDELayer
from .shared_definition import NONLINEARITIES
import numpy as np
import open3d as o3d
import os
from collections import OrderedDict

class ImNet(nn.Module): # develop a transformation based IMNet?
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

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, dim=3, in_features=16, hidden_features=128, 
                 hidden_layers=3,out_features=3, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.dim = dim
        self.in_features = in_features
        self.dimz = dim + in_features

        self.net = []
        self.net.append(SineLayer(self.dimz, hidden_features,  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):


        """
         Args:
          coords: `[batch_size, dim+in_features]` tensor, inputs to decode.
        Returns:
          output through this layer of shape [batch_size, out_features].
        """
        # print("show grad of coords:",coords.requires_grad)
        coords = coords#.requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)

        # print("show shape of output:",output.shape)
        
        return output #, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


# PointAttN: You Only Need Attention for Point Cloud Completion
# https://github.com/ohhhyeahhh/PointAttN
class cross_transformer(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # transformer
    def forward(self, src1, src2, if_act=False):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]

        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)

        src1 = src1.permute(1, 2, 0)

        return src1


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

        assert arch in ["imnet", "vanilla","Siren"]
        if arch == "imnet":
            self.net = ImNet(
                dim=dim,
                in_features=latent_size,
                out_features=dim,
                nf=width,
                nonlinearity=nonlinearity,
            )
        elif arch == "Siren":
            self.net = Siren(
                dim=dim,
                in_features=latent_size,
                hidden_features=128,
                hidden_layers=2,
                out_features=3, outermost_linear=True,
                first_omega_0=30, hidden_omega_0=30.)
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

        # print("show self.divfree:",self.divfree) # False

        def fwd_fn(points):
            """Forward function.

            Where inpt[..., 0], inpt[..., 1], inpt[..., 2] correspond to x, y,
              z and
            out[..., 0], out[..., 1], out[..., 2] correspond to u, v, w
            """
            points_latents = torch.cat((points, latent_vector), axis=-1)  # [batch, num_points, dim + latent_size]
            b, n, d = points_latents.shape

            if self.arch== "Siren":
                res = self.net(points_latents)            
                res = res.reshape([b, n, self.dim])
            else:
                res = self.net(points_latents.reshape([-1, d]))            
                res = res.reshape([b, n, self.dim])

            # print("check grad of 2res:",res.requires_grad) # True
            # print("check shape of 2res:",res.shape) # True
            # exit()
            return res

        if self.divfree:
            # return the curl of the velocity field instead
            self.curl.update_forward_method(fwd_fn)
            _, res_dict = self.curl(points)  # res are the equation residues
            res = torch.cat([res_dict["curl_x"], res_dict["curl_y"], res_dict["curl_z"]],dim=-1,)  # [batch, num_points, dim]            
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

        # precomputed priors
        self.basis_params=None
        self.std_dev_params=None
        self.flow_lat_params=None
        

        # self.scale = nn.Parameter(torch.ones(1))
        

    def add_encoder(self, encoder):
        self.encoder = encoder

    def add_lat_params(self, lat_params, para_name):
        # self.lat_params = lat_params
        setattr(self, para_name, lat_params)

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
        self.latent_seq_bins = torch.cumsum(
            self.latent_seq_weight, dim=1
        )  # [batch, nsteps-1]
        self.latent_seq_bins = torch.cat(
            [torch.zeros([bs, 1], device=dev), self.latent_seq_bins], dim=1
        )  # [batch, nsteps]

        self.latent_updated = True

        return self.latent_seq_bins

  
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


        # print("check grad of latent_val:",latent_val.requires_grad) # False
        # exit()

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
                latent_val, latent_dir, sign = self.latent_at_t(t) # latent_at_t_two_time_step
                # latent_val, latent_dir, sign = self.latent_at_t_two_time_step(t) # latent_at_t_two_time_step
                # # print("show self.scale:",self.scale)
                # print(" latent_val grad:",latent_val.requires_grad) # True
                # exit()
                sign = sign[:, None, None] * self.scale
                if self.symm_dim is None:
                    # print("symm_dim show points grad:",points.requires_grad)
                    flow = self.flow_net(latent_val, points)  # [batch, num_pints, dim]
                else:
                    flow = symmetrize(self.flow_net, latent_val, points, self.symm_dim)
                # Normalize velocity based on time space proportional to latent
                # difference.
                # print("before show flow shape:",flow.shape) #[8, 1024, 3
                # print("see self.latent_seq_len_sum[:, None, None]:",self.latent_seq_len_sum[:, None, None])
                flow *= self.latent_seq_len_sum[:, None, None]
                # print("show flow shape  :",flow.shape)#torch.Size([8, 1024, 3])

                # exit()

                if not self.no_sign_net:
                    sign = self.sign_net(latent_dir)
                flow_signed=flow * sign

                # print("show flow_signed shape:",flow_signed.shape)# torch.Size([8, 1024, 3])
                # exit()
            return flow_signed
        
        else:
            latent_val, latent_dir, sign = self.latent_at_t(t) # latent_at_t_two_time_step
            # latent_val, latent_dir, sign = self.latent_at_t_two_time_step(t) # latent_at_t_two_time_step
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
    def __init__(self, args, shape_mask_clusters=None):
        super(Model,self).__init__()

        self.method = args.solver
        self.conformal = False
        self.lat_dims= args.lat_dims
        self.arch = args.arch
        self.adjoint = args.adjoint
        self.nonlinearity= args.nonlin
        self.no_sign_net= True # try sign net 
        self.deformer_nf= args.deformer_nf

        self.guidance_interpolation= args.guidance_interpolation
        self.coarse_to_fine = args.coarse_to_fine # True # False

        # self.shape_mask_clusters= shape_mask_clusters
        # TODO: try vector field from flowSSM ?



        self.odeint = odeint_adjoint if args.adjoint else odeint_regular
        self.odeint_high_freq = odeint_adjoint if args.adjoint else odeint_regular

        

        self.__timing = torch.from_numpy(
            np.array([0.0, 1.0]).astype("float32")
        ).to(args.device)
        self.return_waypoints = args.return_waypoints
        # return_waypoints: bool, return intermediate waypoints along timing.

        self.use_latent_waypoints = args.use_latent_waypoints
        self.rtol = args.rtol
        self.atol = args.atol
        self.via_hub = args.via_hub
        self.symm_dim = (2 if args.symm else None)

        self.num_part_points = args.num_input_points 
        self.latent_dim=args.lat_dims



        # theta--> theta1_for_transformation + theta_for_deformation 

        

        self.net = torch.nn.ModuleList([])

        self.net.append(
                NeuralFlowModel(
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
                )
        
        # self.net.append(
        #         NeuralFlowModel(
        #         dim=3,
        #         latent_size=self.lat_dims,
        #         f_nlayers=4,
        #         f_width=self.deformer_nf,
        #         s_nlayers=2,
        #         s_width=5,
        #         arch=self.arch,
        #         conformal=self.conformal,
        #         nonlinearity=self.nonlinearity,
        #         no_sign_net=self.no_sign_net,
        #         divfree=False,
        #         symm_dim=(2 if args.symm else None),
        #             )                    
        #         )
        
        # self.net.append(
        #         NeuralFlowModel(
        #         dim=3,
        #         latent_size=self.lat_dims,
        #         f_nlayers=4,
        #         f_width=self.deformer_nf,
        #         s_nlayers=2,
        #         s_width=5,
        #         arch=self.arch,
        #         conformal=self.conformal,
        #         nonlinearity=self.nonlinearity,
        #         no_sign_net=self.no_sign_net,
        #         divfree=False,
        #         symm_dim=(2 if args.symm else None),
        #             )                    
        #         )
        
        self.lvl_detail=len(self.net)
    
       
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


    def basis_transformed(self, refine_theta_en, points, guidance_inter_num=None, shape_name=None):

        """
        
        points: [batch, num_points, dim], the source points

        """
        

        tmp_path= "./temp_folder"

        basis_params= self.net[0].basis_params.to(points.device)
        std_dev_params = self.net[0].std_dev_params.to(points.device)
        # print("show basis_params shape:",basis_params.shape)#  (3072, 64)
        # print("show std_dev_params shape:",std_dev_params.shape)# (64, 1)
        # print("show points shape:",points.shape)# torch.Size([2, 1024, 3])
        std_dev_params = std_dev_params.transpose(1, 0).expand(points.shape[0], -1)  # [batch, dim, latent_size]
        # std_dev_params= std_dev_params.to(points.device)
        # print("show std_dev_params shape:",std_dev_params.shape)# torch.Size([2, 64])
        relative_theta = refine_theta_en[:, 1, :] - refine_theta_en[:, 0, :] 
        interpolated_vectors_pts = []
        interpolated_thetas= []
        zero_vector= torch.zeros_like(relative_theta, device=points.device)
        for i in range(1, guidance_inter_num + 1+1):
            alpha = i / (guidance_inter_num + 1)  # Interpolation factor
            interpolated_vector = zero_vector + alpha * (relative_theta - zero_vector)
            interpolated_vector= interpolated_vector*std_dev_params
            checkReconsPC = points+ torch.matmul(interpolated_vector, basis_params.T).reshape(points.shape[0], self.num_part_points, 3)           
            # print("show checkReconsPC shape:",checkReconsPC.shape)# torch.Size([2, 1024, 3])
            interpolated_vectors_pts.append(checkReconsPC)
            interpolated_thetas.append(interpolated_vector)


        interpolated_vectors_pts = torch.stack(interpolated_vectors_pts)  # Shape: (guidance_inter_num, ...)        
        
        
        #  visualize the interpolated shapes: done
        # interpolated_thetas = torch.stack(interpolated_thetas)  # Shape: (guidance_inter_num, ...)
        # print("show interpolated_vectors_pts shape:",interpolated_vectors_pts.shape)# torch.Size([2, 2, 1024, 3])
        # for idx, vec in enumerate(interpolated_thetas):
        #     print("show vec shape:",vec.shape)# 8, 1024, 3
        #     checkReconsPC = points+ torch.matmul(vec, basis_params.T).reshape(points.shape[0], self.num_part_points, 3)
        #     print("show checkReconsPC shape:",checkReconsPC.shape)# torch.Size([2, 1024, 3])
          
        #     save_name= "interpolated_"+str(idx)+str(shape_name[0])
        #     save_name= os.path.join(tmp_path, save_name)
        #     for idx_1, pts in enumerate(checkReconsPC):
        #         checkReconsPC_pcd = o3d.geometry.PointCloud()
        #         checkReconsPC_pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().numpy())
        #         # checkReconsPC_pcd = self.denormalize_for_inference(checkReconsPC_pcd, self.global_normalization)
        #         o3d.io.write_point_cloud(save_name+"interpolated_"+str(idx_1)+".ply", checkReconsPC_pcd)
            
                                    
                                    # checkReconsPC = self.precomputed_ssm.theta_to_shape_norm(vec*self.theta_std_dev, self.num_nodes)            
                                    # checkReconsPC_pcd = o3d.geometry.PointCloud()
                                    # checkReconsPC_pcd.points = o3d.utility.Vector3dVector(checkReconsPC)
                                    # checkReconsPC_pcd = self.denormalize_for_inference(checkReconsPC_pcd, self.global_normalization)
                                    # o3d.io.write_point_cloud(f"{save_name}.ply", checkReconsPC_pcd)
                                
        # exit()




        # relative_theta= relative_theta*std_dev_params
        # # print("show theta shape:",relative_theta.shape)#  show theta shape: torch.Size([2, 64])
        # trans= torch.matmul(relative_theta, basis_params.T).reshape(points.shape[0], self.num_part_points, 3)
        # # print("show trans shape:",trans.shape)#  ?
        # # points_trans = points + trans
        # points=  points + trans




     
        # data_proj = self.mean + np.matmul(theta.transpose(1, 0), evecs.transpose(1, 0))
        # data_proj = data_proj.reshape(-1, 3)
        #  self.mean= data_proj - np.matmul(theta.transpose(1, 0), evecs.transpose(1, 0))
        # return points
        return interpolated_vectors_pts
    

    def add_lat_params(self, lat_params, para_name):

        for i in range(self.lvl_detail):
            self.net[i].add_lat_params(lat_params,para_name)

        # self.net.add_lat_params(lat_params,para_name)

    # def add_model_params(self, lat_params, para_name):
    #     setattr(self, para_name, lat_params)

    def get_lat_params(self, idx):
        return self.net.get_lat_params(idx)

    def forward(self, points, latent_sequence, pca_guidance_latents=None, shape_name= None):

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
        pca_guidance_latents= pca_guidance_latents.requires_grad_(False)
        # latent_sequence = latent_sequence.requires_grad_(True) # why? should be false?
        # print("show points points shape:",points.shape)# 8, 1024, 3        
        # print("see shape basis_params_evecs:",self.net.basis_params.shape)#  (3072, 64)
        # print("see shape theta_params_std_dev:",self.net.std_dev_params.shape)# (64, 1)
        # Integrate a pre-Computed PCA Basis into the deformation flow field        
        
        
        # print("see latent_sequence val:",latent_sequence)# torch.Size([2, 2, 64])
        # is_nonzero = torch.any(latent_sequence != 0, dim=-1)
        # print("show is_nonzero grad:",is_nonzero)
        # encoded = torch.zeros((latent_sequence.shape[0],latent_sequence.shape[1], latent_sequence.shape[2]), device=points.device)
        # refine_theta_en = torch.zeros_like(latent_sequence)
        # print("show latent_sequence[is_nonzero]:",latent_sequence[is_nonzero])
        # encoded[is_nonzero] = self.latent_en_shared_layers(latent_sequence[is_nonzero])
        # print("show encoded val:",encoded)
        # print("show shape of latent_sequence:",latent_sequence.shape)# torch.Size([2, 2, 10])
        # condition_theta_en = torch.zeros((latent_sequence.shape[0],latent_sequence.shape[1], latent_sequence.shape[2]), device=points.device)
        # condition_theta_en[is_nonzero]=self.theta_condition_head(encoded[is_nonzero])
        # # print("show condition_theta_en grad:",condition_theta_en.shape) #  torch.Size([2, 2, 20])
        # print("see waypoints shape:",waypoints.shape) #  torch.Size([8, 2])
        # print("see waypoints values:",waypoints) #  torch.Size([8, 2])
        # print("show device of waypoints:",points.device)
        # print("show device of refine_theta_en:",refine_theta_en.device) # refine_theta_en, points 


       
        interpo_points_basis_transformed= self.basis_transformed(pca_guidance_latents, points, guidance_inter_num=self.guidance_interpolation, shape_name=shape_name ) # points are source points---> transformed points    
        # print("show shape off interpo_points_basis_transformed:",interpo_points_basis_transformed.shape)# 3, 2, 1024, 3
        interpo_points_basis_transformed= interpo_points_basis_transformed.view(-1, interpo_points_basis_transformed.shape[2], interpo_points_basis_transformed.shape[-1])
        # print("show shape off interpo_points_basis_transformed:",interpo_points_basis_transformed.shape) # 6, 1024, 3
        # torch.Size([3, 8, 1024, 3]) --> torch.Size([24, 1024, 3])
        # exit()
        # # shape: (3, 2, 1024, 3)        
        num_interp= self.guidance_interpolation+1
        relative_flow_theta = latent_sequence[:, 1, :] - latent_sequence[:, 0, :] # relative_theta = refine_theta_en[:, 1, :] - refine_theta_en[:, 0, :] 
        # print("show relative_flow_theta shape:",relative_flow_theta.shape)# torch.Size([2, 10])
        interpolated_vectors_flow = []
        zero_vector= torch.zeros_like(relative_flow_theta, device=points.device)
        # print("show zero_vector shape:",zero_vector.shape)# torch.Size([2, 2, 64])
        for i in range(1, num_interp + 1):
            alpha = i / (num_interp + 1)  # Interpolation factor
            interpolated_vector = zero_vector + alpha * (relative_flow_theta - zero_vector)
            interpolated_vectors_flow.append(interpolated_vector)
        interpolated_vectors_flow = torch.stack(interpolated_vectors_flow)  # Shape: (guidance_inter_num, ...)

     

        
        # TODO: 当我用latent_sequence去转换points 时，在update 函数里也应该把source latent改成latent_sequence
        # print("show points_basis_transformed grad:",points_basis_transformed.requires_grad)# torch.Size([2, 1024, 3])
        # print("show points_basis_transformed shape:",points_basis_transformed.shape)# torch.Size([2, 1024, 3])
        # #vis o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_basis_transformed[0].detach().cpu().numpy())
        # o3d.visualization.draw_geometries([pcd])
        # pcd2= o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(points_basis_transformed[1].detach().cpu().numpy())
        # o3d.visualization.draw_geometries([pcd2])
        # print("show latent_sequence[is_nonzero] shape:",latent_sequence[is_nonzero].shape)# torch.Size([2, 1024, 3])
        # print("show encoded[~is_nonzero]:",encoded[~is_nonzero].shape)# torch.Size([2, 1024, 3])
        # print("show val of encoded[~is_nonzero]:",encoded[~is_nonzero])
        # encoded[~is_nonzero]+=latent_sequence[is_nonzero]
        # print("show latent_sequence shape:",latent_sequence.shape) # torch.Size([2, 2, 10])
        
        is_nonzero = torch.any(latent_sequence != 0, dim=-1)
        # print("see latent_sequence:",latent_sequence.shape)# torch.Size([8, 2, 16])
        # interpolated_latent_vectors=torch.zeros_like(latent_sequence.repeat(num_interp, 1,1))

        # print("show is_nonzero shape:",is_nonzero.shape) # torch.Size([8, 2])
        
        
        


        
        
       
        # print("show is_nonzero :",is_nonzero) 
        # latent_sequence_interpolated = latent_sequence.clone()
        # print("shape of interpo_points_basis_transformed:",interpo_points_basis_transformed.shape)# # shape: (4, 2, 1024, 3)
        
        # use batch operation?
     
        

        print("show shape of interpolated_vectors_flow:",interpolated_vectors_flow.shape)
        # torch.Size([3, 8, 16])
        # interpolated_vectors_flow= interpolated_vectors_flow.view(-1, interpolated_vectors_flow.shape[-1])
        # torch.Size([24, 16])
        
        
        
        # print("show shape of interpolated_vectors_flow:",interpolated_vectors_flow.shape)
        # # torch.Size([24, 16]
        # print("see interpolated_latent_vectors:",interpolated_latent_vectors.shape)# torch.Size([24, 2, 16])
        # print("show is_nonzero_mask shape:",is_nonzero_mask.shape) # torch.Size([24, 2])


        # interpolated_latent_vectors[is_nonzero_mask,:] = interpolated_vectors_flow

        # print("show interpolated_latent_vectors:",interpolated_latent_vectors.shape)
        # print("show val of interpolated_latent_vectors:",interpolated_latent_vectors)


        # points_for_interpo= points.repeat(num_interp, 1, 1)
        points_for_interpo= points.unsqueeze(0).repeat(num_interp, 1, 1,1)
        is_nonzero_mask= is_nonzero.unsqueeze(0).repeat(num_interp,1,1)  # 3,8,2
        _, batch_size, latent_size = interpolated_vectors_flow.shape

        print("show shape of points_for_interpo:",points_for_interpo.shape)# torch.Size([3, 8, 1024, 3])
        print("show shape of interpolated_vectors_flow:",interpolated_vectors_flow.shape) #torch.Size([3, 8, 16])--> torch.Size([3, 8, 2,16])
        print("show shape of latent_sequence:",latent_sequence.shape)#torch.Size([8, 2, 16])
        print("show shape of is_nonzero_mask:",is_nonzero_mask.shape)#3,2,2

        # relative_flow_theta = latent_sequence[:, 1, :] - latent_sequence[:, 0, :]

        interpolated_latent_vectors = torch.zeros(
            (num_interp, batch_size, 2, latent_size),
                    device=interpolated_vectors_flow.device,
                    dtype=interpolated_vectors_flow.dtype
                    )
        
        interpolated_latent_vectors[is_nonzero_mask,:] +=interpolated_vectors_flow.reshape(-1, interpolated_vectors_flow.shape[-1])
        # print("show shape:",a.shape) # torch.Size([6, 16])
        # print("show interpolated_latent_vectors:",interpolated_latent_vectors) # 3,2,2, 16
        # points : 3, 2, 1024, 3
        # exit()
        # 24, 1024, 3

        points_for_interpo= points_for_interpo.reshape(-1, 1024, 3)
        interpolated_latent_vectors= interpolated_latent_vectors.reshape(-1, 2, interpolated_latent_vectors.shape[-1])

        # _ = self.net[0].update_latents(interpolated_latent_vectors)
        # if self.use_latent_waypoints:
        #     timing = waypoints[0]
        # else:
        #     timing = self.timing
        # points_transformed_interp = self.odeint(
        #     self.net[0],
        #     points_for_interpo, # points,
        #     timing,
        #     method=self.method,
        #     rtol=self.rtol,
        #     atol=self.atol,
        #     )            


        # exit()



        # for idx, lat_seq in enumerate(interpolated_vectors_flow):
        #     latent_sequence_interpolated= torch.zeros_like(latent_sequence, device=points.device)
        #     latent_sequence_interpolated[is_nonzero,:] = lat_seq
        #     # print("show grad of latent_sequence_interpolated:",latent_sequence_interpolated.requires_grad)# True            
        #     _ = self.net[0].update_latents(latent_sequence_interpolated)
        #     if self.use_latent_waypoints:
        #         timing = waypoints[0]
        #     else:
        #         timing = self.timing
        #     points_transformed = self.odeint(
        #         self.net[0],
        #         points, # points,
        #         timing,
        #         method=self.method,
        #         rtol=self.rtol,
        #         atol=self.atol,
        #         )            
        #     g_points_transformed.append(points_transformed[-1])
        # g_points_transformed= torch.stack(g_points_transformed)

        # COARSE TO FINE, pyramid structure

        # print("show shape of points_for_interpo:",points_for_interpo.shape)
        # 24, 1024, 3
        # print("show shape of points:",points.shape)
        # # 8, 1024, 3
        # exit()

        if self.coarse_to_fine and self.training :

            points= torch.cat([points_for_interpo,points], dim=0)

            latent_sequence= torch.cat([interpolated_latent_vectors,latent_sequence], dim=0)


            # print("show shape of points:",points.shape) # 8,1024, 3>>> 6+2
            # print("show shape of latent_sequence:",latent_sequence.shape) #>>>  8,2,16
            # exit()


        waypoints = self.net[0].update_latents(latent_sequence)
        if self.use_latent_waypoints:
            timing = waypoints[0]
        else:
            timing = self.timing
        points_transformed = self.odeint(
            self.net[0],
            points, 
            timing,
            method=self.method,
            rtol=self.rtol,
            atol=self.atol,
        )

       
        return points_transformed[-1],  interpo_points_basis_transformed





# self.odeint_high_freq 
# condition_theta_en = torch.zeros((latent_sequence.shape[0],latent_sequence.shape[1], latent_sequence.shape[2]), device=points.device)
# condition_theta_en[is_nonzero]=self.theta_condition_head(encoded[is_nonzero])
# # print("show condition_theta_en grad:",condition_theta_en.shape) #  torch.Size([2, 2, 20])
# condition_theta_en[~is_nonzero]+=encoded[is_nonzero]
# _ = self.net[1].update_latents(condition_theta_en)
# points_transformed_2 = self.odeint_high_freq(
#     self.net[1],
#     points_transformed[-1], # points,
#     timing,
#     method=self.method,
#     rtol=self.rtol,
#     atol=self.atol,
# )
# print("show points_transformed_2 shape:",points_transformed_2[-1].shape)# torch.Size([2, 8, 1024, 3])
# exit()
 # print("show waypoints requires_grad:", waypoints.requires_grad) # True
# print("show points_transformed requires_grad:", points_transformed.requires_grad)# True
# print("show points requires_grad:", points.requires_grad) # True
# print("show timing grad:", timing.requires_grad) # True


# deform_magnitude = torch.norm(points_transformed - points).item()
# print(f"Deformation magnitude: {deform_magnitude}")
# print("show points_transformed shape :", points_transformed.shape)
# # print(f"Flow network grad norm: {torch.norm(next(self.net.flow_net.parameters()).grad).item()}"
# exit()


# deform_abs = torch.mean(
#         torch.norm(points - points_transformed, dim=-1)
#     )
# print("deform_abs deformation!",deform_abs)
# if deform_abs < 1e-6:
#     print("no deformation!")
#     exit()


# print("show points_transformed no -1 shape:",points_transformed.shape)
# shape: torch.Size([2, 8, 1024, 3])

# exit()
# if self.return_waypoints:
#     return points_transformed
#     # print("show points_transformed shape:",points_transformed.shape)
# else:
#     # print("show points_transformed[-1] shape:",points_transformed[-1].shape)# torch.Size([2, 8, 1024, 3])
#     # intermediate= points_transformed[0]
#     # exit()
#     return points_transformed[-1], points_basis_transformed
