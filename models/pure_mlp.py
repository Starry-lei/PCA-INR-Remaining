import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_size = 3           # Just the (x, y, z) of the PCA-reconstructed points
        self.hidden_size = args.latent_dim
        self.output_size = 3          # Residual vector (dx, dy, dz)
        self.num_layers = args.num_layers

        self.mlp = self.build_mlp()

    def build_mlp(self):
        # A simple MLP with ReLU activations
        layers = []
        in_features = self.input_size
        out_features = self.hidden_size

        # First hidden layer
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU(inplace=True))

        # Additional hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(out_features, out_features))
            layers.append(nn.ReLU(inplace=True))

        # Final output layer
        layers.append(nn.Linear(out_features, self.output_size))

        return nn.Sequential(*layers)
    
    def forward(self, points, pca_coeffs=None):
        """
        Args:
            points: Point cloud [batch_size, num_points, 3]
            pca_coeffs: PCA coefficients [batch_size, 1, pca_dim]
                        Provided here, but not used in this unconditioned MLP.
                        If you'd like to incorporate them, 
                        you can concatenate them to points before MLP.
        Returns:
            deformed_points: The original points recovered by adding predicted residuals.
        """
        # points: [B, N, 3]
        batch_size, num_points, _ = points.shape

        # Flatten to [B*N, 3]
        flat_points = points.view(batch_size * num_points, 3)
        
        # If you wish to use pca_coeffs, you could do something like:
        # pca_coeffs = pca_coeffs.expand(-1, num_points, -1)  # [B, N, pca_dim]
        # flat_pca = pca_coeffs.reshape(batch_size * num_points, -1)
        # flat_input = torch.cat([flat_points, flat_pca], dim=-1)
        # residuals = self.mlp(flat_input).view(batch_size, num_points, 3)

        # For unconditioned MLP (no pca_coeffs)
        residuals = self.mlp(flat_points).view(batch_size, num_points, 3)

        deformed_points = points + residuals
        return deformed_points
