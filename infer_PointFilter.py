

from dataset.data_loader_denoising import PCDataset,inferPCDataset,paired_collate_fn
from dataset.data_loader_denoising import PCDValataset
# from dataset.data_loader import CurriculumPCDataset
import argparse
import munch
import datetime
import os
import yaml
import torch
import random
import importlib
import logging
import sys
import torch.optim as optim
import wandb
from tqdm import tqdm
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import open3d as o3d
import torch.nn.functional as F
from utils_ssm.SSM import *
from utils_ssm.evaluation_utils import (
    get_correspondended_vertices,
    get_target_point_cloud,
    get_test_point_cloud,
    save_point_cloud
)


output_dir= "./output_dir"
# wandb.init(project="PCA_latent_denoiser", entity="thesis_lei")

def log_point_clouds_grid( point_clouds, n_rows=2, n_cols=2, step= None):
    """
    Create a 4x6 grid of interactive 3D point clouds
    Args:
        point_clouds: list of point clouds, each with shape (4096, 3)
    """
    # Create HTML table for 4x6 grid layout
    html = "<table style='border-spacing: 10px;'>"
    
    for i in range(n_rows):  # 4 rows
        html += "<tr>"
        for j in range(n_cols):  # 6 columns
            idx = i * n_cols + j
            if idx < len(point_clouds):
                html += "<td style='border: 1px solid gray; padding: 5px;'>"
                # Log each point cloud directly
                wandb.log({f"shape_{idx}": wandb.Object3D(point_clouds[idx])}, step=step)
                html += f"<div style='width: 250px; height: 250px;'></div>"
                html += f"<div style='text-align: center;'>Shape {idx}</div>"
                html += "</td>"
        html += "</tr>"
    html += "</table>"

    wandb.log({"point_clouds_layout": wandb.Html(html)}, step=step)


# the input is PCA reduced 3D point cloud and the corresponding projection parameters, the output is the 3D point cloud before PCA reduction
# I would like to learn a residual prediction model to predict the residual between the PCA reduced 3D point cloud and the original 3D point cloud
# the point cloud in the dataset is already normalized by the bounding box, and all the points in the dataset are ordered, whcih means points at the same index are correspondence
def load_part_names( partNamesPath):
        partNames=[]
        with open(partNamesPath, "r") as f:
            for line in f:
                partNames.append(line.strip())
        print("partNames:", partNames)

        return partNames


# Define helper functions
def load_pretrained_network(args,network_path):
    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    # shape_mask_clusters = dataset_train.mask_clusters
    model = model_module.Model(args,)
    model = model.to(args.device) 
    model.load_state_dict(torch.load(network_path))
    return model

def compute_ssm(corrVertsPath):
    corresponded_train_verts, n_particles = get_correspondended_vertices(None, path=corrVertsPath)
    raw_data_matrix = np.transpose(corresponded_train_verts, (1, 0))
    return SSM(raw_data_matrix)

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def train(args, log_dir=None):
    if torch.cuda.is_available():
        print(f"Total GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No CUDA GPUs are available.")
    
    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    accelerator = Accelerator(
        mixed_precision= 'no', #'fp16',
        gradient_accumulation_steps=1,
        # log_with=args.report_to,
        project_dir=os.path.join(output_dir, "logs"),
        fsdp_plugin=None,
        # even_batches=True,
        kwargs_handlers=[init_handler]
    )
    # LOG.info(accelerator.state)
    num_gpus = accelerator.state.num_processes
    print(f"Number of GPUs being used: {num_gpus}")

    num_epochs = args.epochs

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = args.device


    dataset_infer = inferPCDataset(args, 'infer')
    dataloader_infer = torch.utils.data.DataLoader(dataset_infer, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    

    # model_module = importlib.import_module('.%s' % args.model_name, 'models')
    # # shape_mask_clusters = dataset_train.mask_clusters
    # model = model_module.Model(args,)
    # model = model.to(device) 

    # if hasattr(model_module, 'weights_init'):
    #     model.apply(model_module.weights_init)

    best_val_loss = float('inf')   
    lr = args.lr
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))

    part_name_path= "./dataset/partNames.txt"
    part_names= load_part_names(part_name_path)






    preComputedSSMs=dict()
    preComputedDenoiser=dict()

    added_noise_dim= args.added_noise_dim 

    for cat in part_names:
        precomputed={}
        part_name= cat
        print("show part_name:",part_name)
        # exit()
        # dataset/part_chair_back_pca64/train_corrVerts/corrVerts.npy
        corrVertsPath= os.path.join("./dataset","part_chair_"+part_name+"_pca64","train_corrVerts/corrVerts.npy")
        # exp_shapeNet_chair_armrest/point2ssm/latest_model_weights.pth
        denoiser_network= os.path.join("./exp_shapeNet_chair_"+cat,"point2ssm","latest_model_weights.pth")
        preComputedDenoiser[cat]=load_pretrained_network(args, denoiser_network) 

        # corresponded_train_verts, n_particles = get_correspondended_vertices(None, path=corrVertsPath)
        # raw_data_matrix = np.transpose(corresponded_train_verts, (1, 0))
        # print("show shape of raw_data_matrixL",raw_data_matrix.shape)
        # precomputed_ssm = SSM(raw_data_matrix)
        preComputedSSMs[cat]=compute_ssm(corrVertsPath)
        print(f"Completed processing for part: {part_name}")

 
    mu=0
    sigma=1


    recons_path= os.path.join(args.work_dir, "recons")
    noised_path= os.path.join(args.work_dir, "noised")
    denoised_path= os.path.join(args.work_dir, "deoised")

    if not os.path.exists(recons_path):
        os.makedirs(recons_path, exist_ok=True)
        os.makedirs(noised_path, exist_ok=True)
        os.makedirs(denoised_path, exist_ok=True)



    with torch.no_grad():
        tqdm_train_loader = tqdm(dataloader_infer, desc=f"Refine point cloud...")
        for shape_idx, data_tensors in enumerate(tqdm_train_loader):

            labels, thetas = data_tensors # one shape information
            # check the PCA reconstruction:
            print("show label shapes:",labels.shape)# torch.Size([1, 4])
            print("show thetas shapes:",thetas.shape)# torch.Size([1, 4, 64])
            print("show cur shape part labels:",labels)


            
        

            for idx, label in enumerate(labels):


                shape_part_pca_recons_points=[]
                shape_part_pca_recons_points_noised=[]
                shape_part_pca_recons_points_denoised=[]
                shape_pca_recons_pc= o3d.geometry.PointCloud()
                shape_pca_recons_pc_noised= o3d.geometry.PointCloud()
                shape_pca_recons_pc_de_noised= o3d.geometry.PointCloud()

                print("show only idx:",idx)

                # checkReconsPC_noised_pcd_normalized_points_pca= o3d.geometry.PointCloud()
                denoised_pc_pcd= o3d.geometry.PointCloud()

                for cat, theta_vector in zip(label, thetas[idx]):

                    if cat==0:
                        checkReconsPC= np.zeros((1024,3))
                        checkReconsPC_noised=np.zeros((1024,3))
                        denoised_pc=np.zeros((1024,3))
                    else:
                        cat_name = part_names[cat-1]
                        # print("show labels:",cat_name)# back
                        theta_vector_norm=theta_vector.unsqueeze(-1)                    
                        # print("show theta shape:",theta_vector.shape)# torch.Size([64,1]) 
                        theta_variance = preComputedSSMs[cat_name].get_variance_num_modes(num_modes=args.lat_dims)
                        # print("show theta_std_dev_cur_cat shape:",theta_variance.shape)#  (64,)  
                        theta_std_dev= np.sqrt(theta_variance.reshape(-1,1))
                        theta_std_dev= torch.from_numpy(theta_std_dev).float()
                        # print("show theta_std_dev shape:",theta_std_dev.shape)#  (64,1) 
                        theta_vector_unnorm= theta_vector_norm*theta_std_dev
                        basis_evecs= preComputedSSMs[cat_name].modes_norm[:, :args.lat_dims]
                        # print("show basis_evecs shape:",basis_evecs.shape)
                        theta_vector_unnorm= theta_vector_unnorm.cpu().numpy()
                        checkReconsPC = preComputedSSMs[cat_name].theta_to_shape_norm(theta_vector_unnorm, args.lat_dims)


                        # add noise and denoise
                        theta_variance_more = preComputedSSMs[cat_name].get_variance_num_modes(num_modes=args.lat_dims+args.added_noise_dim)
                        theta_variance_more=theta_variance_more.reshape(-1, 1)
                        theta_std_variance_more = np.sqrt(theta_variance_more)

                        noise = np.random.normal(loc=mu, scale=sigma, size=(args.added_noise_dim,1))
                        theta_normalized_noised = np.concatenate([theta_vector_norm,noise],axis=0 )
                        checkReconsPC_noised = preComputedSSMs[cat_name].theta_to_shape_norm(theta_normalized_noised*theta_std_variance_more, args.lat_dims+args.added_noise_dim)
                        # normalize the noised point cloud
                        checkReconsPC_noised_pcd= o3d.geometry.PointCloud()
                        checkReconsPC_noised_pcd.points=o3d.utility.Vector3dVector(checkReconsPC_noised)

                        checkReconsPC_noised_pcd_normalized= preComputedSSMs[cat_name].normalize_bounding_box(preComputedSSMs[cat_name].global_normalization, checkReconsPC_noised_pcd)
                        checkReconsPC_noised_pcd_normalized_points= np.array(checkReconsPC_noised_pcd_normalized.points)
                        print("show shape of checkReconsPC_noised_pcd_normalized_points:", checkReconsPC_noised_pcd_normalized_points.shape)
                        # checkReconsPC_noised_pcd_normalized_points_pca.points= o3d.utility.Vector3dVector(checkReconsPC_noised_pcd_normalized_points)
                        # o3d.io.write_point_cloud("./testREconsNoised.ply", checkReconsPC_noised_pcd_normalized_points_pca)
                        checkReconsPC_noised_pcd_normalized_points_tensor=torch.from_numpy(checkReconsPC_noised_pcd_normalized_points).float() 
                        checkReconsPC_noised_pcd_normalized_points_tensor= checkReconsPC_noised_pcd_normalized_points_tensor.unsqueeze(0).to(args.device)
                        denoised_pc = preComputedDenoiser[cat_name](checkReconsPC_noised_pcd_normalized_points_tensor, is_training=False)
                        denoised_pc_numpy= denoised_pc.squeeze(0).cpu().numpy()# (1024,3)
                        denoised_pc_pcd.points=o3d.utility.Vector3dVector(denoised_pc_numpy)
                        denoised_pc_pcd_un_normalzied= preComputedSSMs[cat_name].denormalize_for_inference(denoised_pc_pcd,preComputedSSMs[cat_name].global_normalization )
                        denoised_pc= np.array(denoised_pc_pcd_un_normalzied.points)
                        print("checkReconsPC shape:", checkReconsPC.shape)# (1024, 3), numpy


                    shape_part_pca_recons_points.append(checkReconsPC)
                    shape_part_pca_recons_points_noised.append(checkReconsPC_noised)
                    shape_part_pca_recons_points_denoised.append(denoised_pc)
                  


            shape_part_pca_recons_points=np.concatenate(shape_part_pca_recons_points, axis=0)
            shape_pca_recons_pc.points= o3d.utility.Vector3dVector(shape_part_pca_recons_points)
            
            shape_part_pca_recons_points_noised=np.concatenate(shape_part_pca_recons_points_noised, axis=0)
            shape_pca_recons_pc_noised.points= o3d.utility.Vector3dVector(shape_part_pca_recons_points_noised)

            shape_part_pca_recons_points_denoised=np.concatenate(shape_part_pca_recons_points_denoised, axis=0)
            shape_pca_recons_pc_de_noised.points= o3d.utility.Vector3dVector(shape_part_pca_recons_points_denoised)

            pcd_name= "shape_"+str(shape_idx)+".ply"
            recons_pcd_name_save_path= os.path.join(recons_path, pcd_name)
            noised_pcd_save_path= os.path.join(noised_path, pcd_name)
            denosied_save_path= os.path.join(denoised_path, pcd_name)
            o3d.io.write_point_cloud(recons_pcd_name_save_path, shape_pca_recons_pc)
            o3d.io.write_point_cloud(noised_pcd_save_path, shape_pca_recons_pc_noised)
            o3d.io.write_point_cloud(denosied_save_path, shape_pca_recons_pc_de_noised)



                    

      





            # pca_noised_recon= pca_noised_recon.to(device)
            # pca_input= pca_input.to(device)
            # gt_5k_points= gt_5k_points.to(device)
            # pred = model(pca_noised_recon,is_training=False)
            # print("show shape of pred:",pred.shape)
            # exit()


def main():
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)

    arg= parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    print_time = datetime.datetime.now().isoformat()[:19]

    if args.load_model:
        exp_name = os.path.basename(os.path.dirname(args.load_model))
        # print("checking exp_name:",exp_name)
        log_dir = os.path.dirname(args.load_model)
        # print("checking log_dir:",log_dir)
    else:
        if 'encoder' in args:
            exp_name = args.model_name+'_'+args.encoder
        else:
            exp_name = args.model_name
        exp_name += '_'+print_time.replace(':',"-")
        log_dir = os.path.join(args.work_dir, args.dataset, exp_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print("log_dir:",log_dir)
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(os.path.join(log_dir, 'train.log')),
                                                        logging.StreamHandler(sys.stdout)])
    train(args, log_dir=log_dir)


    #test() visualize the result

if __name__ == '__main__':
    # conda activate freereg strucNet/ freereg
    # python train_PointFilter.py -c cfgs/config.yaml

    # python train_PointFilter.py -c cfgs/config_part_back.yaml
    # python train_PointFilter.py -c cfgs/config_part_seat.yaml
    # python train_PointFilter.py -c cfgs/config_part_armrest.yaml

    #  inference
    # python infer_PointFilter.py -c cfgs/config_part_armrest.yaml

    # python train_PointFilter.py -c cfgs/infer_config.yaml
    # python infer_PointFilter.py -c cfgs/infer_config.yaml
    main()




