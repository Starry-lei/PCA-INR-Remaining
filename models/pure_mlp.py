import torch
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_size = 3           # Just the (x, y, z) of the PCA-reconstructed points
        self.hidden_size = args.hidden_size
        self.output_size = 3          # Residual vector (dx, dy, dz)
        self.num_layers = args.num_layers
        self.num_points= 1024

        self.mlp = self.build_mlp()

    def build_mlp(self):
        # A simple MLP with ReLU activations
        layers = []
        in_features = self.input_size* self.num_points
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
    
    def forward(self, points, pca_coeffs):
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

        points = points.permute(0, 2, 1)  # shape: [B, 3, 1024]

        # Flatten to [B*N, 3]
        flat_points = points.view(batch_size , num_points*3)
        

        residuals = self.mlp(flat_points).view(batch_size, num_points, 3)

        deformed_points = points + residuals
        return deformed_points
