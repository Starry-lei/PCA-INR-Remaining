
# from pycarus.learning.models.siren import SIREN
import torch.nn as nn
import torch
import numpy as np



class PiGANMappingNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, depth=3):
        super().__init__()
        layers = []

        if depth == 0:
            layers.extend([nn.Linear(dim_in, dim_out)]) 
        else:
            layers.extend([nn.Linear(dim_in, dim_hidden), nn.LeakyReLU(0.2, inplace=True)]) 
            nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)
            for i in range(depth-1):
                layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU(0.2, inplace=True)]) 
                nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)
            layers.extend([nn.Linear(dim_hidden, dim_out)]) 
            nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)

        self.net = nn.Sequential(*layers)
        
        with torch.no_grad():
            self.net[-1].weight *= 0.25

    def forward(self, x):
        x = self.net(x)
        gamma, beta = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return gamma, beta
    


class ImplicitMLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                omega_0=1, w_norm=False, activation="relu", omega_uniform=False,
                film_conditioning=False, concat_conditioning=0,
                init_method={"weights": 'basic', "bias": "zero"}):
        super().__init__()

        self.omega_0 = nn.Parameter(torch.ones(out_features, requires_grad=False)*omega_0,requires_grad=False)

        if omega_uniform:
            omegas = torch.sort(torch.rand(out_features, requires_grad=False)*omega_0/in_features)
            self.omega_0 = (nn.Parameter(omegas[0],requires_grad=False))

        self.in_features = in_features
        self.out_features = out_features
        self.film_conditioning = film_conditioning
        self.concat_conditioning = concat_conditioning
        if concat_conditioning:
            self.in_features += concat_conditioning 

        self.linear = nn.Linear(self.in_features, out_features, bias=bias)

        if activation == "relu":
            self.activation = nn.LeakyReLU(negative_slope=0.02)
        if activation == "sine":
            self.activation = torch.sin
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        if activation == "tanh":
            self.activation = nn.Tanh()
        if activation == "none":
            self.activation = self.return_input

        with torch.no_grad():
            self.init_weights(init_method)
            if w_norm:
                self.linear = torch.nn.utils.weight_norm(self.linear)

    def return_input(self, input):
        return input

    def init_weights(self, init_method):
        with torch.no_grad():
            if init_method["weights"] == "basic":
                # Values taken from IM-NET
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
            if init_method["weights"] == "kaiming_in":
                nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.02)
                with torch.no_grad():
                    self.linear.weight /= self.omega_0
            if init_method["weights"] == "siren":
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0.abs().mean(), 
                                            np.sqrt(6 / self.in_features) / self.omega_0.abs().mean())
            if init_method["weights"] == "siren_omega":
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / init_method["omega"], 
                                            np.sqrt(6 / self.in_features) / init_method["omega"])
            if init_method["weights"] == "siren_first":
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)   
            if init_method["weights"] == "none":
                pass
            
            if init_method["bias"] == "zero":
                nn.init.constant_(self.linear.bias, 0)
            if init_method["bias"] == "polar":
                self.linear.bias.uniform_(0, 2*np.pi)
            if init_method["bias"] == "none":
                pass  



    def forward(self, layer_input, z=None, gamma=None, beta=None, delta=None, progress=None):
        if self.concat_conditioning:
            if z.shape[1] !=  layer_input.shape[1]: 
                z = z.repeat(1, layer_input.shape[1], 1)
            layer_input = torch.cat((layer_input, z), dim=-1)

        if self.film_conditioning:
            self.feat_multiplier = gamma[:, :,  :self.out_features] + self.omega_0
            self.feat_bias = beta[:, :, :self.out_features]

        else:
            self.feat_multiplier = self.omega_0
            self.feat_bias = 0

        output = self.activation((self.feat_multiplier * self.linear(layer_input)) + self.feat_bias)
        if delta is not None:
            output = output * delta
        return output



class CondSIREN(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                num_layers=3, num_hidden=256, num_mapping_layers=2, pca_dim=None,
                first_omega_0=30, hidden_omega_0=30):
        super().__init__()

        self.num_mapping_layers = num_mapping_layers

        self.mapping_net = PiGANMappingNetwork(pca_dim, num_hidden, num_hidden*2, depth = self.num_mapping_layers)

        self.net = []


        self.net.append(ImplicitMLPLayer(in_features, num_hidden, bias=True,
                        omega_0=first_omega_0, w_norm=False, activation="sine", 
                        film_conditioning=True, concat_conditioning=0,
                        init_method={"weights": 'siren_first', "bias": "polar"}))
        for i in range(num_layers-1):
            self.net.append(ImplicitMLPLayer(num_hidden, num_hidden, bias=True,
                            omega_0=hidden_omega_0, w_norm=False, activation="sine", 
                            film_conditioning=True, concat_conditioning=0,
                            init_method={"weights": 'siren', "bias": "polar"}))
            
        self.net.append(ImplicitMLPLayer(num_hidden, out_features, bias=True,
                omega_0=1, w_norm=False, activation="none", 
                film_conditioning=False, concat_conditioning=0,
                init_method={"weights": 'siren_omega', "omega":30, "bias": "none"}))
        self.net = nn.Sequential(*self.net)



    def forward(self, x, x_cond):

        gamma, beta = self.mapping_net(x_cond)

        # print("see  gamma shape", gamma.shape)# torch.Size([8, 1024, 512])
        # print("see  beta shape", beta.shape) # torch.Size([8, 1024, 512])

        output = self.net[0](x, gamma=gamma, beta=beta)

        # print("see  1output shape", output.shape)# torch.Size([8, 1024, 512])

        for layer in self.net[1:]:
            if layer.film_conditioning:
                output = layer(output, gamma=gamma, beta=beta)
            else:
                output = layer(output)

        # print("see  2output shape", output.shape)# shape torch.Size([8, 1024, 3])
        return output
    


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.num_layers = args.num_layers
        self.pca_dim = args.pca_dim
        self.mlp = self.build_mlp(args)
    

    def build_mlp(self, args) -> CondSIREN:

        mlp = CondSIREN(
            in_features=3,
            out_features=3,
            bias=True,
            num_layers=4,
            num_hidden=512,  
            num_mapping_layers=args.num_mapping_layers,
            pca_dim=args.pca_dim,            
        )


        return mlp
    
    def forward(self, points, pca_coeffs):
        """
        Args:
            points: Point cloud [batch_size, num_points, 3]
            pca_coeffs: PCA coefficients [batch_size, 1, pca_dim]
                       (projections onto eigenvectors)
        Returns:
            Point residuals with shape [batch_size, num_points, 3]
        """
        batch_size, num_points, _ = points.shape
        pca_coeffs = pca_coeffs.expand(-1, num_points, -1)
        residuals = self.mlp(points,pca_coeffs)  
        # print("see residuals", residuals.shape) # see residuals torch.Size([8, 1024, 3])
        # exit()
        deformed_points = points + residuals
        return deformed_points
        