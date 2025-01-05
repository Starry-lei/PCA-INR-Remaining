

import sys
sys.path.append('..')
import torch
import torch.utils.data as data
import numpy as np
import os
import open3d as o3d
from glob import glob
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from utils_ssm.SSM import *
from utils_ssm.evaluation_utils import (
    get_correspondended_vertices,
    get_target_point_cloud,
    get_test_point_cloud,
    save_point_cloud
)
from sklearn.cluster import KMeans


def load_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)


def saveCorrespondence(data_files, corrVertsPath):
    corres_verts = []
    corr_files =sorted(glob(os.path.join(data_files,'*.txt'))) # txt, particles # output_half #output
    
    assert len(corr_files) != 0, "No correspondence files found"

    # print("show corr_files:",corr_files)
    # exit()
    ref_mat = load_points(corr_files[0])
    for vert in ref_mat:
        corres_verts.append([vert])
    for idx,  corr_file in enumerate(corr_files):
        if idx == 0:
            continue
        print("corr_file", corr_file)
        file_mat = load_points(corr_file)
        for idx,  each in enumerate(file_mat):
            corres_verts[idx].append(each)
    corres_verts = np.array(corres_verts)
    print("corres_verts shape:",corres_verts.shape)
    np.save(corrVertsPath, corres_verts)
    print("corres_verts saved")


# use mean shape for normalization

def normalize_bounding_box(global_normalization, pca_input):
    # culate the axis-aligned bounding box
    aabb_center = global_normalization[0]
    scale_factor= global_normalization[1]
    pca_input.translate(-aabb_center)    
    # pca_recon.translate(-aabb_center)
    # Calculate extents of the bounding box
    # Apply the scaling
    pca_input.scale(scale_factor, center=[0, 0, 0])  # Scale around the new origin    
    # pca_recon.scale(scale_factor, center=[0, 0, 0])
    return pca_input


def normalize_mean_shape(global_normalization, mean_shape):
    aabb_center = global_normalization[0]
    scale_factor= global_normalization[1]
    mean_shape.translate(-aabb_center)    
    mean_shape.scale(scale_factor, center=[0, 0, 0])  # Scale around the new origin    
    return mean_shape

# perform SSM computation heere, integrate the precomputed PCA basis into the network
def cluster_points(points, method='kmeans', n_clusters=4, eps=0.5, min_samples=5):
    """
    Cluster 3D points from a shape using different clustering methods.

    Parameters:
    points: numpy array of shape (n_points, 3) containing x,y,z coordinates
    method: str, clustering method ('kmeans', 'dbscan', or 'spectral')
    n_clusters: int, number of clusters for KMeans and Spectral clustering
    eps: float, maximum distance between points for DBSCAN
    min_samples: int, minimum number of samples in a neighborhood for DBSCAN

    Returns:
    labels: cluster labels for each point
    """
    # Standardize the points
    # scaler = StandardScaler()
    # points_scaled = scaler.fit_transform(points)
    points_scaled = points

    # Perform clustering
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == 'dbscan':
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError("Method must be one of: 'kmeans', 'dbscan', 'spectral'")

    labels = clusterer.fit_predict(points_scaled)

    return labels


# part level dataset
class PCDataset(data.Dataset):
    def __init__(self, args, set_type='val'):

        # run dataloader for train, then val set

        self.data_path = os.path.join(args.dataset, set_type+'_set/', "pca_input/") 
        self.num_points= args.num_input_points # 1024 
        self.num_nodes = args.lat_dims # 64
        self.n_clusters= args.n_clusters # 4
        
        self.corrVertsPath= os.path.join(args.dataset, set_type+'_corrVerts/', "corrVerts.npy") 
        self.preComputedPCA= os.path.join(args.dataset, set_type+'_corrVerts/', "preComputedPCA.npy") 
        if not os.path.exists(os.path.dirname(self.corrVertsPath)):
            os.makedirs(os.path.dirname(self.corrVertsPath))
        
        self.data_files= sorted(os.listdir(self.data_path))

        self.data_length = len(self.data_files)



        # print("show (np.load(self.corrVertsPath)).shape[1]:",(np.load(self.corrVertsPath)).shape[1])
        # print("show len(self.data_files):",len(self.data_files))
        # exit()
        if not os.path.exists(self.corrVertsPath):
            saveCorrespondence(self.data_path, self.corrVertsPath)
        # elif len(self.data_files) != (np.load(self.corrVertsPath)).shape[0]:
        #     saveCorrespondence(self.data_path, self.corrVertsPath)
        corresponded_train_verts, n_particles = get_correspondended_vertices(None, path=self.corrVertsPath)
        self.raw_data_matrix = np.transpose(corresponded_train_verts, (1, 0))
        self.mean_shape_flat = np.mean(self.raw_data_matrix, 0)
        self.mean_shape = self.mean_shape_flat.reshape(self.num_points, 3)


        # perform clustering over the mean shape
        # clustering the mean shape:
        labels = cluster_points(self.mean_shape, method='kmeans', n_clusters=self.n_clusters)    
        unique_labels, label_counts = np.unique(labels, return_counts=True)
        print("unique_labels:",unique_labels)# [0 1 2 3]
        print("label_counts:", label_counts) # 282, 227，287，228
        points_with_labels = np.column_stack((self.mean_shape, labels))

        combined_path = "chair_leg_points_with_labels_"+str(self.n_clusters)+".txt"
        combined_file_path = os.path.join(args.dataset, combined_path)
        if not os.path.exists(combined_file_path):
            np.savetxt(combined_file_path, points_with_labels)

        # mask_cluster_0 = points_with_labels[:, 3] == 0

        self.mask_clusters= [ points_with_labels[:, 3] == i for i in range(self.n_clusters)]

        # print("shape of mask_clusters:",len(mask_clusters)) # 4    
        # print("mask_cluster_0:",mask_cluster_0.shape) # (1024,)
        # # num_points_cluster_0 = np.sum(mask_cluster_0) # 282
        # # print("num_points_cluster_0:",num_points_cluster_0)
        # exit()


        masked_pc = points_with_labels[points_with_labels[:, 3] == labels[0]]
        points = masked_pc[:, :3]


        # exit()
        self.mean_shape_pcd = o3d.geometry.PointCloud()
        self.mean_shape_pcd.points = o3d.utility.Vector3dVector(self.mean_shape)
        self.global_normalization = self.get_scale_factor( self.mean_shape_pcd)
        self.normalized_mean_shape_pcd = normalize_mean_shape(self.global_normalization, self.mean_shape_pcd)

        # normalize all input shapes
        print(f"Loading {set_type} data")
        print(self.data_path)# dataset/part_chair_leg_pca64
        self.pca_input_points_sets = []
        self.pca_recon_points_sets = []
        self.pca_scales_sets = []
        self.pca_theta_sets= []
        self.names = []


        for file in self.data_files:
            if file.endswith(".txt"):
                data_pca_input = self.data_path + file   
                pca_input_points = np.asarray(np.loadtxt(data_pca_input))
            else:
                continue
            # normalization here # TODO: change this
            pca_input_points_pcd = o3d.geometry.PointCloud()
            pca_input_points_pcd.points = o3d.utility.Vector3dVector(pca_input_points)
            pca_input_points_pcd = normalize_bounding_box(self.global_normalization ,pca_input_points_pcd)
            self.pca_input_points_sets.append(np.asarray(pca_input_points_pcd.points))
            self.names.append(file.replace(".txt", ""))

        self.pca_input_points_sets_pca= np.array(self.pca_input_points_sets)
        self.pca_input_points_sets_pca_flat= self.pca_input_points_sets_pca.reshape(self.pca_input_points_sets_pca.shape[0], -1)
        
        
        self.precomputed_ssm = SSM(self.pca_input_points_sets_pca_flat)


        self.theta_variance = self.precomputed_ssm.get_variance_num_modes(num_modes=self.num_nodes)
        self.theta_variance = self.theta_variance.reshape(-1, 1)
        self.theta_std_dev = np.sqrt(self.theta_variance)

        
        


        self.basis_evecs = self.precomputed_ssm.modes_norm[:, :self.num_nodes] if self.num_nodes else self.precomputed_ssm.modes_norm
        
        precomputed_pca = {
            "theta_std_dev": self.theta_std_dev,
            "basis_evecs": self.basis_evecs
        }
        if not os.path.exists(self.preComputedPCA):
            np.save(self.preComputedPCA, precomputed_pca)
                
        print("show shape of self.theta_std_dev: ",self.theta_std_dev.shape)#  (64, 1)
        print("show shape of self.basis_evecs:", self.basis_evecs.shape)# (3072, 64)


        # data_proj = self.mean + np.matmul(theta.transpose(1, 0), evecs.transpose(1, 0))
        # data_proj = data_proj.reshape(-1, 3)
        # exit()
        num_interpolations = 5

        max_loss= 0
        self.names_loss ={}

        for index, normalzied_input in enumerate(self.pca_input_points_sets):
            # print("see shape of normalzied_input:",normalzied_input.shape)# (1024, 3)

            # pcd_view = o3d.geometry.PointCloud()
            # pcd_view.points = o3d.utility.Vector3dVector(normalzied_input)

            loss= F.mse_loss(torch.tensor(normalzied_input), torch.tensor(self.mean_shape), reduction='mean')
       
            if loss.item() > max_loss:
                max_loss= loss.item()
            self.names_loss[self.names[index]]= loss.item()




            theta_vector = self.precomputed_ssm.get_theta(normalzied_input, self.num_nodes)            
            # whiting/sphere
            theta_normalized = theta_vector / self.theta_std_dev
            self.pca_theta_sets.append(theta_normalized)
            # checkReconsPC = self.precomputed_ssm.theta_to_shape_norm(theta_vector, self.num_nodes)

            # checkReconsPC_interpolated
            # interpolated_vectors = []
            # zero_vector= np.zeros_like(theta_normalized)
            # for i in range(1, num_interpolations + 1):
            #     alpha = i / (num_interpolations + 1)  # Interpolation factor
            #     interpolated_vector = zero_vector + alpha * (theta_normalized - zero_vector)
            #     interpolated_vectors.append(interpolated_vector)

            # # Convert list to numpy array for better readability
            # interpolated_vectors = np.array(interpolated_vectors)
            # for idx, vec in enumerate(interpolated_vectors, 1):
            #     print(f"Interpolated Vector {idx}: {vec}")
            #     checkReconsPC = self.precomputed_ssm.theta_to_shape_norm(vec*self.theta_std_dev, self.num_nodes)
            #     save_name= self.names[index]+"_interpolated_"+str(idx)
            #     checkReconsPC_pcd = o3d.geometry.PointCloud()
            #     checkReconsPC_pcd.points = o3d.utility.Vector3dVector(checkReconsPC)
            #     checkReconsPC_pcd = self.denormalize_for_inference(checkReconsPC_pcd, self.global_normalization)
            #     o3d.io.write_point_cloud(f"{save_name}.ply", checkReconsPC_pcd)
            # exit()




            # pcd_view2 = o3d.geometry.PointCloud()
            # pcd_view2.points = o3d.utility.Vector3dVector(checkReconsPC)
            # o3d.visualization.draw_geometries([pcd_view,pcd_view2])
            # exit()

        losses = np.array(list(self.names_loss.values())).reshape(-1, 1)
        # Define percentile thresholds for 40/40/20 split
        p40 = np.percentile(losses, 40)  # Easy threshold
        p80 = np.percentile(losses, 80)  # Medium threshold
        easy_samples = {}
        medium_samples = {}
        hard_samples = {}

        for name, loss in self.names_loss.items():
            if loss <= p40:
                easy_samples[name] = loss
            elif loss <= p80:
                medium_samples[name] = loss
            else:
                hard_samples[name] = loss

        # Print statistics
        print(f"Total samples: {len(self.names_loss)}")
        print(f"Easy samples: {len(easy_samples)} ({len(easy_samples)/len(self.names_loss)*100:.1f}%)")
        print(f"Medium samples: {len(medium_samples)} ({len(medium_samples)/len(self.names_loss)*100:.1f}%)")
        print(f"Hard samples: {len(hard_samples)} ({len(hard_samples)/len(self.names_loss)*100:.1f}%)")
        # Print threshold values
        print(f"\nThresholds:")
        print(f"Easy-Medium threshold (40th percentile): {p40:.6f}")
        print(f"Medium-Hard threshold (80th percentile): {p80:.6f}")
        print(f"Max loss: {max_loss:.6f}")


        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.hist(losses, bins=50, alpha=0.7)
        # plt.axvline(p40, color='g', linestyle='--', label='40th percentile (Easy-Medium)')
        # plt.axvline(p80, color='r', linestyle='--', label='80th percentile (Medium-Hard)')
        # plt.xlabel('Loss Values')
        # plt.ylabel('Frequency')
        # plt.title('Distribution of Deformation Losses (40/40/20 split)')
        # plt.legend()
        # plt.show()
        # exit()

        # print("see shape of pca_input_points_sets:",self.pca_input_points_sets_pca.shape)        
        # print("see shape of pca_input_points_sets_array_flat:",self.pca_input_points_sets_pca_flat.shape)       
        # reduant_mean= np.mean(self.pca_input_points_sets_array_flat, 0)
        # reduant_mean_mean_shape = reduant_mean.reshape(self.num_points, 3)
        # self.reduant_mean_mean_shape_shape_pcd = o3d.geometry.PointCloud()
        # self.reduant_mean_mean_shape_shape_pcd.points = o3d.utility.Vector3dVector(reduant_mean_mean_shape)
        # o3d.visualization.draw_geometries([self.reduant_mean_mean_shape_shape_pcd, self.normalized_mean_shape_pcd])
        # o3d.io.write_point_cloud("1filename.ply", self.reduant_mean_mean_shape_shape_pcd)
        # o3d.io.write_point_cloud("2filename.ply", self.normalized_mean_shape_pcd)
        # exit()




    def get_scale_factor(self, mean_shape_pcd):

        aabb = mean_shape_pcd.get_axis_aligned_bounding_box()    
        # Get the center of the bounding box
        aabb_center = aabb.get_center()   


        # Calculate extents of the bounding box
        aabb_extent = aabb.get_extent()    
        # Calculate the norm of the extents vector
        norm_extent = np.linalg.norm(aabb_extent)    
        # Determine the scaling factor as the reciprocal of the norm of the extents
        scale_factor = 1 / norm_extent  



        return (aabb_center, scale_factor)

    def denormalize_for_inference(self, pcd, scale_factors):
    
        aabb_center= scale_factors[0]
        scale_factor = scale_factors[1]
        # Inverse the scaling
        pcd.scale(1/scale_factor, center=[0, 0, 0])
        # Inverse the translation
        pcd= pcd.translate(aabb_center)
        
        return pcd
    
    def get_global_normalization(self):
        return self.global_normalization


    def __getitem__(self, index):
        pca_input = self.pca_input_points_sets[index]
        name = self.names[index]
        # pca_recon = self.pca_recon_points_sets[index]
        pca_theta = self.pca_theta_sets[index]
        

        pca_input = torch.from_numpy(pca_input).float()
        # pca_recon = torch.from_numpy(pca_recon).float()
        pca_theta = torch.from_numpy(pca_theta).float()
        pca_theta = pca_theta.T

        shape_idx= torch.tensor(index, dtype=torch.long)

        # print(pca_input.shape)
        # print(pca_recon.shape)
        # print(pca_theta.shape)
        # torch.Size([1024, 3])
        # torch.Size([1024, 3])
        # torch.Size([1,64])
        # exit()



        return shape_idx, name, pca_theta, pca_input

    def __len__(self):
        return len(self.pca_input_points_sets)
    

class PCDValataset(data.Dataset):
    def __init__(self, args, set_type='val', global_normalization=None, mean_shape=None, precomputed_ssm=None):

        # run dataloader for train, then val set

        self.data_path = os.path.join(args.dataset, set_type+'_set/', "pca_input/") 
        self.num_points= args.num_input_points # 1024 
        self.num_nodes = args.lat_dims # 64

        self.global_normalization = global_normalization
        self.mean_shape_normalized = mean_shape
        self.data_files= sorted(os.listdir(self.data_path))


        print(f"Loading {set_type} data")
        print(self.data_path)# dataset/part_chair_leg_pca64
        self.pca_input_points_sets = []
        self.pca_recon_points_sets = []
        self.pca_scales_sets = []
        self.pca_theta_sets= []
        self.names = []


        for file in self.data_files:
            if file.endswith(".txt"):
                data_pca_input = self.data_path + file   
                pca_input_points = np.asarray(np.loadtxt(data_pca_input))
            else:
                continue
            # normalization here # TODO: change this
            pca_input_points_pcd = o3d.geometry.PointCloud()
            pca_input_points_pcd.points = o3d.utility.Vector3dVector(pca_input_points)
            pca_input_points_pcd = normalize_bounding_box(self.global_normalization ,pca_input_points_pcd)
            self.pca_input_points_sets.append(np.asarray(pca_input_points_pcd.points))
            self.names.append(file.replace(".txt", ""))

    
        self.precomputed_ssm = precomputed_ssm


        self.theta_variance = self.precomputed_ssm.get_variance_num_modes(num_modes=self.num_nodes)
        self.theta_variance = self.theta_variance.reshape(-1, 1)
        self.theta_std_dev = np.sqrt(self.theta_variance)
        self.basis_evecs = self.precomputed_ssm.modes_norm[:, :self.num_nodes] if self.num_nodes else self.precomputed_ssm.modes_norm

        print("show shape of self.theta_std_dev: ",self.theta_std_dev.shape)#  (64, 1)
        print("show shape of self.basis_evecs:", self.basis_evecs.shape)# (3072, 64)


        # data_proj = self.mean + np.matmul(theta.transpose(1, 0), evecs.transpose(1, 0))
        # data_proj = data_proj.reshape(-1, 3)
        # exit()

        for index, normalzied_input in enumerate(self.pca_input_points_sets):
            # print("see shape of normalzied_input:",normalzied_input.shape)# (1024, 3)

            # pcd_view = o3d.geometry.PointCloud()
            # pcd_view.points = o3d.utility.Vector3dVector(normalzied_input)

            theta_vector = self.precomputed_ssm.get_theta(normalzied_input, self.num_nodes)            
            # whiting/sphere
            theta_normalized = theta_vector / self.theta_std_dev
            self.pca_theta_sets.append(theta_normalized)
            checkReconsPC = self.precomputed_ssm.theta_to_shape_norm(theta_vector, self.num_nodes)

            # pcd_view2 = o3d.geometry.PointCloud()
            # pcd_view2.points = o3d.utility.Vector3dVector(checkReconsPC)
            # o3d.visualization.draw_geometries([pcd_view,pcd_view2])
            # exit()


        # print("see shape of pca_input_points_sets:",self.pca_input_points_sets_pca.shape)        
        # print("see shape of pca_input_points_sets_array_flat:",self.pca_input_points_sets_pca_flat.shape)       
        # reduant_mean= np.mean(self.pca_input_points_sets_array_flat, 0)
        # reduant_mean_mean_shape = reduant_mean.reshape(self.num_points, 3)
        # self.reduant_mean_mean_shape_shape_pcd = o3d.geometry.PointCloud()
        # self.reduant_mean_mean_shape_shape_pcd.points = o3d.utility.Vector3dVector(reduant_mean_mean_shape)
        # o3d.visualization.draw_geometries([self.reduant_mean_mean_shape_shape_pcd, self.normalized_mean_shape_pcd])
        # o3d.io.write_point_cloud("1filename.ply", self.reduant_mean_mean_shape_shape_pcd)
        # o3d.io.write_point_cloud("2filename.ply", self.normalized_mean_shape_pcd)
        # exit()




    def get_scale_factor(self, mean_shape_pcd):

        aabb = mean_shape_pcd.get_axis_aligned_bounding_box()    
        # Get the center of the bounding box
        aabb_center = aabb.get_center()   


        # Calculate extents of the bounding box
        aabb_extent = aabb.get_extent()    
        # Calculate the norm of the extents vector
        norm_extent = np.linalg.norm(aabb_extent)    
        # Determine the scaling factor as the reciprocal of the norm of the extents
        scale_factor = 1 / norm_extent  



        return (aabb_center, scale_factor)

    def denormalize_for_inference(self, pcd, scale_factors):
    
        aabb_center= scale_factors[0]
        scale_factor = scale_factors[1]
        # Inverse the scaling
        pcd.scale(1/scale_factor, center=[0, 0, 0])
        # Inverse the translation
        pcd= pcd.translate(aabb_center)
        
        return pcd
    
    def get_global_normalization(self):
        return self.global_normalization


    def __getitem__(self, index):
        pca_input = self.pca_input_points_sets[index]
        name = self.names[index]
        # pca_recon = self.pca_recon_points_sets[index]
        pca_theta = self.pca_theta_sets[index]
        

        pca_input = torch.from_numpy(pca_input).float()
        # pca_recon = torch.from_numpy(pca_recon).float()
        pca_theta = torch.from_numpy(pca_theta).float()
        pca_theta = pca_theta.T

        # print(pca_input.shape)
        # print(pca_recon.shape)
        # print(pca_theta.shape)
        # torch.Size([1024, 3])
        # torch.Size([1024, 3])
        # torch.Size([1,64])
        # exit()



        return name, pca_theta, pca_input

    def __len__(self):
        return len(self.pca_input_points_sets)


class CurriculumPCDataset(PCDataset):
    def __init__(self, args, set_type='train', phase='easy'):
        super().__init__(args, set_type)
        
        # Calculate percentile thresholds for 40/40/20 split
        self.losses = np.array(list(self.names_loss.values()))
        self.p40 = np.percentile(self.losses, 40)  # Easy threshold
        self.p80 = np.percentile(self.losses, 80)  # Medium threshold

        self.data_length = len(self.names_loss)

        # Create a list of (index, name, loss) tuples to preserve original indexing
        indexed_data = [(i, name, loss) for i, (name, loss) in enumerate(self.names_loss.items())]
        
        # Create indices for each difficulty level while preserving original indices
        self.easy_indices = [idx for idx, _, loss in indexed_data if loss <= self.p40]
        self.medium_indices = [idx for idx, _, loss in indexed_data if self.p40 < loss <= self.p80]
        self.hard_indices = [idx for idx, _, loss in indexed_data if loss > self.p80]
        
        # Verify the indices
        # print(f"Easy indices range: {(self.easy_indices)}")
        # print(f"Medium indices range: {(self.medium_indices)} ")
        # print(f"Hard indices range: {(self.hard_indices)}")
        # exit()
        self.phase = phase
        self.update_active_indices()
    
    def update_active_indices(self):
        """Update active indices based on current phase"""
        if self.phase == 'easy':
            self.active_indices = self.easy_indices
        elif self.phase == 'medium':
            self.active_indices = self.easy_indices + self.medium_indices
        elif self.phase == 'hard':
            self.active_indices = self.easy_indices + self.medium_indices + self.hard_indices
        else:
            raise ValueError(f"Invalid phase: {self.phase}")
    
    def set_phase(self, phase):
        """Update the training phase"""
        self.phase = phase
        self.update_active_indices()
    
    def __getitem__(self, index):
        """Get item from active indices only"""
        actual_index = self.active_indices[index]
        return super().__getitem__(actual_index)
    
    def __len__(self):
        """Return length of active dataset"""
        return len(self.active_indices)





class InferenceDataset(data.Dataset):
    def __init__(self, recon_folder_path):
        self.recon_points_sets = []
        self.pca_rep_sets = []
        self.scale_factors_sets = []
        self.names = []
        
        for file in sorted(os.listdir(recon_folder_path)):
            if not file.endswith(".txt"):
                continue
                
            try:
                # Load reconstructed points
                recon_path = os.path.join(recon_folder_path, file)
                pcd_recon = np.loadtxt(recon_path)
                
                # Load corresponding PCA representation if available
                rep_path = recon_path.replace("pcd_recon", "pca_rep")
                pca_rep = np.loadtxt(rep_path) if os.path.exists(rep_path) else None
                
                # Normalize the reconstructed points
                norm_recon, scale_factors = normalize_for_inference(pcd_recon)
                
                # Store data
                self.recon_points_sets.append(norm_recon)
                self.pca_rep_sets.append(pca_rep)
                self.scale_factors_sets.append(scale_factors)
                self.names.append(os.path.splitext(file)[0])
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    def __getitem__(self, index):
        return {
            'pcd_recon': torch.from_numpy(self.recon_points_sets[index]).float(),
            'pca_rep': torch.from_numpy(self.pca_rep_sets[index]).float() if self.pca_rep_sets[index] is not None else None,
            'name': self.names[index],
            'scale_factors': self.scale_factors_sets[index]
        }
    
    def __len__(self):
        return len(self.names)


# part level dataset
class shapelvlDataset(data.Dataset):
    pass