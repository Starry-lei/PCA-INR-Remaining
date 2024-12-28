

import sys
sys.path.append('..')
import torch
import torch.utils.data as data
import numpy as np
import os
import open3d as o3d



# use mean shape for normalization

def normalize_bounding_box(global_normalization, pca_input, pca_recon):
    # culate the axis-aligned bounding box
    aabb_center = global_normalization[0]
    scale_factor= global_normalization[1]
    pca_input.translate(-aabb_center)    
    pca_recon.translate(-aabb_center)
    # Calculate extents of the bounding box
    # Apply the scaling
    pca_input.scale(scale_factor, center=[0, 0, 0])  # Scale around the new origin    
    pca_recon.scale(scale_factor, center=[0, 0, 0])
    return pca_input, pca_recon


def normalize_mean_shape(global_normalization, mean_shape):
    aabb_center = global_normalization[0]
    scale_factor= global_normalization[1]
    mean_shape.translate(-aabb_center)    
    mean_shape.scale(scale_factor, center=[0, 0, 0])  # Scale around the new origin    
    return mean_shape




# part level dataset
class PCDataset(data.Dataset):
    def __init__(self, args, set_type='val'):

        self.data_path = os.path.join(args.dataset, set_type+'_set/', "pca_input/") 

        self.mean_shape_path = os.path.join(args.dataset, set_type+'_set/', "chair_leg_mean.txt")

        self.mean_shape = np.loadtxt(self.mean_shape_path)
        self.mean_shape_pcd = o3d.geometry.PointCloud()
        self.mean_shape_pcd.points = o3d.utility.Vector3dVector(self.mean_shape)

        self.global_normalization = self.get_scale_factor( self.mean_shape_pcd)

        self.normalized_mean_shape_pcd =normalize_mean_shape(self.global_normalization, self.mean_shape_pcd)
     

        print(f"Loading {set_type} data")
        print(self.data_path)# dataset/part_chair_leg_pca64
        self.pca_input_points_sets = []
        self.pca_recon_points_sets = []
        self.pca_scales_sets = []
        self.pca_rep_sets= []
        self.names = []


        for file in sorted(os.listdir(self.data_path)):
            if file.endswith(".txt"):
                data_pca_input = self.data_path + file
                data_pca_recon = data_pca_input.replace("pca_input", "pca_recon")
                data_pca_rep = data_pca_input.replace("pca_input", "pca_rep") # TODO: change this
                pca_input_points = np.asarray(np.loadtxt(data_pca_input))
                pca_recon_points= np.asarray(np.loadtxt(data_pca_recon))
                pca_rep=  np.asarray(np.loadtxt(data_pca_rep))

               
                assert pca_input_points.shape == pca_recon_points.shape
   
            else:
                continue
            # normalization here # TODO: change this

            pca_input_points_pcd = o3d.geometry.PointCloud()
            pca_input_points_pcd.points = o3d.utility.Vector3dVector(pca_input_points)
            

            pca_recon_points_pcd = o3d.geometry.PointCloud()
            pca_recon_points_pcd.points = o3d.utility.Vector3dVector(pca_recon_points)


            pca_input_points_pcd, pca_recon_points_pcd = normalize_bounding_box(self.global_normalization ,pca_input_points_pcd, pca_recon_points_pcd)
 
            self.pca_input_points_sets.append(np.asarray(pca_input_points_pcd.points))
            self.pca_recon_points_sets.append(np.asarray(pca_recon_points_pcd.points))
            self.pca_rep_sets.append(pca_rep)
            self.names.append(file.replace(".txt", ""))



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
        pca_recon = self.pca_recon_points_sets[index]
        pca_rep = self.pca_rep_sets[index]
        

        pca_input = torch.from_numpy(pca_input).float()
        pca_recon = torch.from_numpy(pca_recon).float()
        pca_rep = torch.from_numpy(pca_rep).float()
        pca_rep = pca_rep.unsqueeze(0)

        # print(pca_input.shape)
        # print(pca_recon.shape)
        # print(pca_rep.shape)
        # torch.Size([1024, 3])
        # torch.Size([1024, 3])
        # torch.Size([1,64])
        # exit()



        return pca_recon, pca_rep, pca_input

    def __len__(self):
        return len(self.pca_input_points_sets)
    




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