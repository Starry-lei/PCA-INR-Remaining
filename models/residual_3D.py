'''Define basic blocks
'''

import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F
from models.coord_models import PE, MLP, Siren, GaborNet, MultiscaleBACON
from models.dgcnn_order import DGCNN_encoder_order

from models.dgcnn import DGCNN_encoder



class OrderedPointConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Use 1D convolution since points are ordered
        return self.conv1d(x.transpose(1, 2)).transpose(1, 2)

class Model(nn.Module):
    def __init__(self, args):
        """
        A pure MLP (fully connected) to process each point [x, y, z].
        hidden_dim: number of hidden units (can be increased for more capacity).
        """
        super().__init__()
        # Example: 3 -> hidden_dim -> hidden_dim -> 3
        self.hidden_dim = 128
        self.pca_rep_in_features=64

        self.in_features= 3
        self.num_layers= 3

        self.pca_rep_emb= self.pca_rep_en_mlp(self.hidden_dim)
        self.pca_recon_en= self.pca_recon_en_mlp(self.hidden_dim) # replace it with dgcnn

        self.encoder = DGCNN_encoder(self.hidden_dim)

        self.pca_input_displancement= self.build_de_mlp(self.hidden_dim*2)


    def build_de_mlp(self, latent_dim):
        # A simple MLP with ReLU activations
        layers = []
        in_features = latent_dim 
        out_features = self.in_features
        # First hidden layer
        layers.append(nn.Linear(in_features, in_features))
        # layers.append(nn.LayerNorm(out_features))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(in_features, in_features))
            # layers.append(nn.LayerNorm(out_features))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(in_features, out_features))

        return nn.Sequential(*layers)
    

    def pca_rep_en_mlp(self, latent_dim):
        # A simple MLP with ReLU activations
        layers = []
        in_features = self.pca_rep_in_features
        out_features = latent_dim
        # First hidden layer
        layers.append(nn.Linear(in_features, out_features))
        # layers.append(nn.LayerNorm(out_features))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(out_features, out_features))
            # layers.append(nn.LayerNorm(out_features))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))
        # Final output layer
        layers.append(nn.Linear(out_features, out_features))

        return nn.Sequential(*layers)

    def pca_recon_en_mlp(self, latent_dim):
        # A simple MLP with ReLU activations
        layers = []
        in_features = self.in_features
        out_features = latent_dim
        # First hidden layer
        layers.append(nn.Linear(in_features, out_features))
        # layers.append(nn.LayerNorm(out_features))
        layers.append(nn.ReLU())
        # Additional hidden layers
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(out_features, out_features))
            # layers.append(nn.LayerNorm(out_features))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(0.2))
        # Final output layer
        layers.append(nn.Linear(out_features, out_features))

        return nn.Sequential(*layers)


    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            print(f"Number of NaN values: {torch.isnan(tensor).sum().item()}")
            print(f"Tensor shape: {tensor.shape}")
            print(f"Min value: {tensor[~torch.isnan(tensor)].min().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
            print(f"Max value: {tensor[~torch.isnan(tensor)].max().item() if (~torch.isnan(tensor)).any() else 'all NaN'}")
            return True
        return False

    def forward(self,pca_recon_displancement, pca_rep):
        """
        Args:
            points: [B, N, 3]
        Returns:
            output: [B, N, 3]
        """
        
        
        batch_size, num_points, _ = pca_recon_displancement.shape  # [B, N, 3]


        pca_rep_Emb= self.pca_rep_emb(pca_rep)
        pca_rep_Emb= pca_rep_Emb.expand(-1, num_points, -1)

        

        # pca_recon_displancement_emb= self.pca_recon_en(pca_recon_displancement) # replace it with dgcnn
        global_fea, pca_recon_displancement_emb= self.encoder(pca_recon_displancement) # replace it with dgcnn



        
        recon_shape_cond= torch.cat([pca_recon_displancement_emb,pca_rep_Emb], dim=-1)

        # print("see pca_rep_Emb shape:",pca_rep_Emb.shape)
        # print("see pca_recon_displancement_emb shape:",pca_recon_displancement_emb.shape)
        # print("see recon_shape_cond shape:",recon_shape_cond.shape)
        # self.check_nan(mean_shape_emb, "mean_shape_emb")        
        # self.check_nan(pca_recon_displancement_emb, "pca_recon_displancement_emb")
        # global_feature, pca_recon_displancement_emb=  self.encoder(pca_recon_displancement)
        # self.check_nan(mean_shape_cond, "mean_shape_cond")

        refine_displacement= self.pca_input_displancement(recon_shape_cond)

        # exit()

        return refine_displacement
 
