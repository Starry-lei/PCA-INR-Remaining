import torch
import torch.nn as nn
import torch.nn.functional as F

# class Model(nn.Module):

#     def __init__(self, args):
#         super(Model, self).__init__()
#         self.input_size = 3           # Just the (x, y, z) of the PCA-reconstructed points
#         self.hidden_size = args.hidden_size
#         self.output_size = 3          # Residual vector (dx, dy, dz)
#         self.num_layers = args.num_layers
#         self.num_points= 1024

#         self.mlp = self.build_mlp_linear(self.hidden_size, self.input_size)



#     def build_mlp_1dcnn(self, hidden_size, input_size):

#         # input_size
#         # A simple MLP with ReLU activations
#         mlp_layers = [
#         nn.Conv1d(input_size, hidden_size, 1),
#         # nn.BatchNorm1d(hidden_size),     
#         nn.GELU(),
#         nn.Conv1d(hidden_size, hidden_size * 2, 1),
#         # nn.BatchNorm1d(hidden_size * 2),     
#         nn.GELU(),
#         nn.Conv1d(hidden_size * 2, input_size, 1),]
#         return nn.Sequential(*mlp_layers)
    

#     def build_mlp_linear(self, hidden_size, input_size):

#         # input_size
#         # A simple MLP with ReLU activations
#         mlp_layers = [
#         nn.Linear(input_size, hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, hidden_size * 2),
#         nn.ReLU(),
#         nn.Linear(hidden_size * 2, hidden_size * 2),
#         nn.ReLU(),
#         nn.Linear(hidden_size * 2, hidden_size),
#         nn.ReLU(),
#         nn.Linear(hidden_size, hidden_size*0.5),
#         nn.ReLU(),
#         nn.Linear( hidden_size*0.5, 3),
        
#         ]
#         return nn.Sequential(*mlp_layers)

    

#     # def build_mlp(self):
#     #     # A simple MLP with ReLU activations
#     #     layers = []
#     #     in_features = self.input_size* self.num_points
#     #     out_features = self.hidden_size

#     #     # First hidden layer
#     #     layers.append(nn.Linear(in_features, out_features))
#     #     layers.append(nn.ReLU(inplace=True))

#     #     # Additional hidden layers
#     #     for _ in range(self.num_layers - 2):
#     #         layers.append(nn.Linear(out_features, out_features))
#     #         layers.append(nn.ReLU(inplace=True))

#     #     # Final output layer
#     #     layers.append(nn.Linear(out_features, self.output_size))

#     #     return nn.Sequential(*layers)



    
#     def forward(self, points, pca_coeffs):
#         """
#         Args:
#             points: Point cloud [batch_size, num_points, 3]
#             pca_coeffs: PCA coefficients [batch_size, 1, pca_dim]
#                         Provided here, but not used in this unconditioned MLP.
#                         If you'd like to incorporate them, 
#                         you can concatenate them to points before MLP.
#         Returns:
#             deformed_points: The original points recovered by adding predicted residuals.
#         """
#         # points: [B, N, 3]
#         batch_size, num_points, _ = points.shape

#         points_p = points.permute(0, 2, 1)  # shape: [B, 3, 1024]

#         # Pass through the MLP (Conv1d stack)
#         points_p = self.mlp(points_p)       # shape: [B, hidden_size, 1024]

#         # print("show shape of points:",points.shape)

#         # exit()
    
#         residuals = points_p.permute(0, 2, 1)  # shape: [B, 1024, 3]
#         deformed_points = points + residuals
#         return deformed_points

class Model(nn.Module):
    def __init__(self, args):
        """
        A pure MLP (fully connected) to process each point [x, y, z].
        hidden_dim: number of hidden units (can be increased for more capacity).
        """
        super().__init__()
        # Example: 3 -> hidden_dim -> hidden_dim -> 3
        hidden_dim = args.hidden_size # 256
        self.fc1 = nn.Linear(3, int(hidden_dim*0.5))
        self.fc2 = nn.Linear(int(hidden_dim*0.5), hidden_dim)


        # self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.sa1 = cross_transformer(hidden_dim, hidden_dim)



        self.fc5 = nn.Linear(hidden_dim, int(hidden_dim*0.5))
        self.fc6 = nn.Linear(int(hidden_dim*0.5), 3)

        self.sa1 = cross_transformer(hidden_dim, hidden_dim)

    def forward(self, points, pca_coeffs):
        """
        Args:
            points: [B, N, 3]
        Returns:
            output: [B, N, 3]
        """
        batch_size, num_points, _ = points.shape  # [B, N, 3]

        # (1) Flatten to [B*N, 3]
        x = points.view(batch_size, num_points, 3)

        # (2) Pass through MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # c = F.relu(self.fc4(x))

        print("see x shape:" ,x.shape)

        x= self.sa1(x,x)
        x = F.relu(self.fc5(x))
        x = self.fc6(x)  # -> [B*N, 3]

        # (3) Reshape back to [B, N, 3]
        residuals = x.view(batch_size, num_points, 3)
        # residuals = points_p.permute(0, 2, 1)  # shape: [B, 1024, 3]
        deformed_points = points + residuals
        return deformed_points
        # return x

# Example usage:
# if __name__ == "__main__":
#     model = Model(hidden_dim=128)
#     # Dummy input: batch_size=8, each with 1024 points in 3D
#     points = torch.randn(8, 1024, 3)

#     out = model(points)
#     print("Input shape: ", points.shape)  # [8, 1024, 3]
#     print("Output shape:", out.shape)     # [8, 1024, 3]



class Attention_Module(nn.Module):
    def __init__(self, latent_dim, num_output):
        super(Attention_Module, self).__init__()
        self.num_output = num_output
        self.latent_dim = latent_dim

        self.sa1 = cross_transformer(self.latent_dim,self.num_output)
        self.sa2 = cross_transformer(self.num_output,self.num_output)
        # self.sa3 = cross_transformer(self.num_output,self.num_output)

       
        # Final projection to residuals [B, N, latent_dim] -> [B, N, 3]
        self.residual_proj = nn.Linear(num_output, 3)


    def forward(self, x):

        x = self.sa1(x,x)
        x = self.sa2(x,x)
        # x = self.sa3(x,x)

        residuals = self.residual_proj(x)  # [B, N, 3]
        return residuals

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
    
