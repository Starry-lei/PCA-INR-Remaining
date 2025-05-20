

from dataset.data_loader_denoising import PCDataset,RandomPairSampler,paired_collate_fn
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
from sklearn.neighbors import NearestNeighbors
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


output_dir= "./output_dir"
wandb.init(project="PCA_latent_denoiser", entity="thesis_lei")

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


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def normalize_and_add_curvature_based_noise_tensor(points_tensor, 
                                                   k=20, 
                                                   noise_std_min=0.005, 
                                                   noise_std_max=0.02, 
                                                   scale_factor=15000):
    """
    For each point cloud in the batch:
      1. Normalize the points by subtracting the center and dividing by the bounding sphere's radius.
      2. Add curvature-based Gaussian noise. The base noise standard deviation is chosen 
         randomly between noise_std_min and noise_std_max (which are fractions of the 
         bounding sphere's radius). Then, each point’s noise is modulated by its local curvature.
    
    Parameters:
        points_tensor (torch.Tensor): Input tensor of shape [b, 2048, 3].
        k (int): Number of nearest neighbors for curvature estimation.
        noise_std_min (float): Minimum fraction for base noise standard deviation.
        noise_std_max (float): Maximum fraction for base noise standard deviation.
        scale_factor (float): Factor to amplify the curvature values.
    
    Returns:
        torch.Tensor: Noisy point clouds of shape [b, 2048, 3] (still normalized).
    """
    b, n, d = points_tensor.shape
    assert d == 3, "Input tensor must be of shape [b, N, 3]."
    
    noisy_batch = []
    
    # Process each point cloud in the batch separately
    for i in range(b):
        # Convert the point cloud to a NumPy array
        points = points_tensor[i].cpu().numpy()  # shape: [2048, 3]
        
        # --- Normalization Step ---
        # Compute the center of the point cloud and subtract it
        center = points.mean(axis=0)
        points_centered = points - center
        
        # Compute the bounding sphere's radius (max Euclidean distance from the center)
        distances = np.linalg.norm(points_centered, axis=1)
        radius = distances.max()
        
        # Normalize the point cloud so that its bounding sphere has radius 1
        points_normalized = points_centered / radius
        
        # --- Noise Injection Step ---
        # Choose a base noise standard deviation randomly (as a fraction of the bounding sphere)
        # Since the cloud is normalized, the base noise is directly in [0.005, 0.02]
        base_noise = random.uniform(noise_std_min, noise_std_max)
        
        # For curvature-based noise, we modulate the noise per point using the local curvature.
        # Find the k nearest neighbors for each point (using the normalized points)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points_normalized)
        _, indices = nbrs.kneighbors(points_normalized)
        
        # Prepare an array to store the noise sigma for each point
        noise_sigmas = np.zeros(points_normalized.shape[0])
        
        for j, neighbors in enumerate(indices):
            # Extract the local neighborhood points
            local_pts = points_normalized[neighbors]  # shape: [k, 3]
            # Compute the 3x3 covariance matrix of the local neighborhood
            cov = np.cov(local_pts.T)
            # Compute eigenvalues (which np.linalg.eigh returns in ascending order)
            eigenvalues, _ = np.linalg.eigh(cov)
            # Use the smallest eigenvalue as a proxy for local flatness (or inversely, curvature)
            curvature = eigenvalues[0]
            # print("curvature:",curvature)
            # Map curvature to noise amplitude:
            # Higher curvature leads to a larger noise standard deviation.
            noise_sigmas[j] = base_noise * (1 + curvature * scale_factor)
        
        # Generate noise for each point based on its computed sigma
        noise = np.array([np.random.randn(3) * sigma for sigma in noise_sigmas])
        # Add the noise to the normalized points
        noisy_points = points_normalized + noise
        
        noisy_batch.append(noisy_points)
    
    # Stack the processed point clouds back into a tensor of shape [b, 2048, 3]
    noisy_batch = np.stack(noisy_batch, axis=0)
    noisy_tensor = torch.tensor(noisy_batch, dtype=points_tensor.dtype, device=points_tensor.device)
    
    return noisy_tensor



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

    dataset_train = PCDataset(args, 'train')

    best_model_weights_path= os.path.join(log_dir, 'best_model_weights.pth')
    latest_model_weights_path=os.path.join(log_dir, 'latest_model_weights.pth')


    

    
    dataset_val= PCDValataset(args, 'val', global_normalization=dataset_train.global_normalization, mean_shape=dataset_train.normalized_mean_shape_pcd, precomputed_ssm=dataset_train.precomputed_ssm)

    # exit()
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.workers))
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
    
    
    logging.info('Length of train dataset:%d', len(dataloader_train))
    logging.info('Length of validation dataset:%d', len(dataloader_val))

    dataloader_train = accelerator.prepare_data_loader(dataloader_train)
    dataloader_val = accelerator.prepare_data_loader(dataloader_val)

    # exit()
    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = args.device

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    

    shape_mask_clusters = dataset_train.mask_clusters
    model = model_module.Model(args,)

    
    # model = model_module.Model(args, shape_mask_clusters=shape_mask_clusters, mean_shape_point_labels= dataset_train.mean_shape_point_labels)
    # try transformer model used in partSSM
    model = model.to(device) 

    if hasattr(model_module, 'weights_init'):
        model.apply(model_module.weights_init)

    best_val_loss = float('inf')   
    lr = args.lr
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))

    
    optimizer = getattr(optim, args.optimizer)  
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min",factor=0.1, patience=10, verbose=True, min_lr=1e-6)

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    the_pca_mean_shape = torch.from_numpy(np.array(dataset_train.normalized_mean_shape_pcd.points)).float().to(device)
    the_pca_mean_shape = the_pca_mean_shape.unsqueeze(0)
    print("shape of the_pca_mean_shape:", the_pca_mean_shape.shape) # shape of the_pca_mean_shape: torch.Size([1,1024, 3])
    


    # load params:
    load_params=  True # True
    if load_params:
        path= "latest_model_weights.pth" # latest_model_weights # best_model_weights
        model.load_state_dict(torch.load(path))

    criterion_mse = torch.nn.MSELoss(reduction='none') # hybrid_loss with chamfer_loss # reduction='none': reduction='sum'
    # criterion_chamfer = calc_cd2



    global_step = 0

    noise_std_min=0.005
    noise_std_max=0.015


    # TODO:
    # adjust the number of used from 20->16, 
    # add some reg or loss to improve the denoised point cloud quality
    # we can still try chamfer loss with GT 5k points, because the network will always output ordered point cloud!

    for epoch in range(num_epochs):
        
        epoch_idx=epoch+1
        model.train()
        total_loss = 0
        tqdm_train_loader = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")

        # print(f"Epoch {epoch}, Phase: {dataset_train.phase}, Active samples: {len(dataset_train)}")
        
        for data_tensors  in tqdm_train_loader:

            shape_idx, name, gt_5k_points, pca_recon, pca_input, pca_noised_recon = data_tensors
            pca_noised_recon= pca_noised_recon.to(device)
            pca_recon= pca_recon.to(device)
            pca_input= pca_input.to(device)
            gt_5k_points= gt_5k_points.to(device)

            # print("show points shape of gt_5k_points:",gt_5k_points.shape)
            # # torch.Size([4, 5000, 3])

            pca_recon_noised_new = normalize_and_add_curvature_based_noise_tensor(pca_recon, k=20, noise_std_min=noise_std_min, noise_std_max=noise_std_max, scale_factor=10000)

            # print("show shape of pca_recon_noised_new: ",pca_recon_noised_new.shape)
            # # torch.Size([16, 1024, 3])
            # print("show shape of pca_input: ",pca_input.shape)
            # #  torch.Size([16, 1024, 3])
            # exit()

            # pred, loss_mse = model(pca_noised_recon, gt=gt_5k_points)
            # pred, loss_mse = model(pca_recon_noised_new, gt=pca_input)

            pred = model(pca_recon_noised_new)
            loss_mse= criterion_mse(pred,pca_input)

            loss= loss_mse.mean()
            # print("show loss:",loss)
            # exit()
            if epoch_idx % 20 == 0:
                predis_pred= pred[:4].detach().cpu().numpy()
                print("show shape of predis_pred:",predis_pred.shape)# 4， 1024， 3
                predis_pred[:,:, [1, 2]] = predis_pred[:,:, [2, 1]]
                log_point_clouds_grid(predis_pred, step=global_step)
               
            print("see loss_mse:",loss.item())
            optimizer.zero_grad()
            # loss.backward()
            accelerator.backward(loss)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()    

            wandb.log({"train_loss": loss.item()}, step=global_step)# , step=global_step
            tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")


        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches= len(dataloader_val)
            # print("num_batches:", num_batches)# 86
            for shape_idx, name, gt_5k_points, pca_recon, pca_input, pca_noised_recon in dataloader_val:
                pca_recon= pca_recon.to(device)

                # print("show shape of pca_recon:",pca_recon.shape)
                # # torch.Size([32, 3])
                # exit()
                pca_noised_recon= pca_noised_recon.to(device)
                pca_input= pca_input.to(device)
                gt_5k_points= gt_5k_points.to(device)

                pca_recon_noised_new = normalize_and_add_curvature_based_noise_tensor(pca_recon, k=20, noise_std_min=noise_std_min, noise_std_max=noise_std_max, scale_factor=10000)

                # pred, loss_mse = model(pca_recon_noised_new, gt=pca_input)
                pred = model(pca_recon_noised_new)
                loss_mse= criterion_mse(pred,pca_input)

                loss= loss_mse.mean()
                # loss = loss_mse
                val_loss += loss.item()
            val_loss /= num_batches


        wandb.log({"val_loss": val_loss,}, step=global_step)# , step=global_step
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_weights_path)
            wandb.log({"best_val_loss": best_val_loss}, step=global_step)


        global_step += 1
        torch.save(model.state_dict(), latest_model_weights_path)


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


if __name__ == '__main__':

    # conda activate freereg strucNet/ freereg
    # python train_PointFilter.py -c cfgs/config.yaml # for leg


    # python train_PointFilter.py -c cfgs/config_part_back.yaml
    # python train_PointFilter.py -c cfgs/config_part_seat.yaml
    # python train_PointFilter.py -c cfgs/infer_config_eva_cosine_decay
   
    main()

