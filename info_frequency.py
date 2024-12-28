

from dataset.data_loader import PCDataset
import argparse
import munch
import datetime
import os
import yaml
import torch
import random
import importlib
import pywt
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

output_dir= "./output_dir"
# wandb.init(project="PCA_Remainig", entity="thesis_lei")
# PCA-Conditioned Residual SIREN MLP

# the input is PCA reduced 3D point cloud and the corresponding projection parameters, the output is the 3D point cloud before PCA reduction
# I would like to learn a residual prediction model to predict the residual between the PCA reduced 3D point cloud and the original 3D point cloud
# the point cloud in the dataset is already normalized by the bounding box, and all the points in the dataset are ordered, whcih means points at the same index are correspondence


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def wavelet_transform_tensor(tensor, wavelet='db1', level=None):
    """
    Apply wavelet transformation to a tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        wavelet (str): Wavelet type to use (default: 'db1')
        level (int): Decomposition level (default: None, which means maximum possible level)
    
    Returns:
        list: Coefficients from wavelet transformation
    """
    # Move tensor to CPU and convert to numpy
    if tensor.is_cuda:
        tensor = tensor.cpu()
    np_array = tensor.numpy()
    
    # Process each sample and channel independently
    batch_size, channels, dim = np_array.shape
    coeffs_batch = []
    
    for b in range(batch_size):
        coeffs_channels = []
        for c in range(channels):
            # Apply 2D wavelet transform
            coeffs = pywt.wavedec2(np_array[b, c], wavelet, level=level)
            coeffs_channels.append(coeffs)
        coeffs_batch.append(coeffs_channels)
    
    return coeffs_batch

def inverse_wavelet_transform_tensor(coeffs_batch, wavelet='db1', original_shape=None):
    """
    Apply inverse wavelet transformation to coefficients.
    
    Args:
        coeffs_batch (list): List of wavelet coefficients
        wavelet (str): Wavelet type used (default: 'db1')
        original_shape (tuple): Shape of the original tensor (batch_size, channels, height, width)
    
    Returns:
        torch.Tensor: Reconstructed tensor
    """
    batch_size = len(coeffs_batch)
    channels = len(coeffs_batch[0])
    reconstructed = np.zeros((batch_size, channels, *original_shape[-2:]))
    
    for b in range(batch_size):
        for c in range(channels):
            # Apply inverse 2D wavelet transform
            reconstructed[b, c] = pywt.waverec2(coeffs_batch[b][c], wavelet)
    
    return torch.from_numpy(reconstructed)



def train(args):


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
    dataset_val = PCDataset(args, 'val')

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=False, num_workers=int(args.workers))
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
    model = model_module.Model(args)
    model = model.to(device) # PCA residual MLP

    if hasattr(model_module, 'weights_init'):
        model.apply(model_module.weights_init)

     
    best_val_loss = float('inf')   
    lr = args.lr
    betas = args.betas.split(',')
    betas = (float(betas[0].strip()), float(betas[1].strip()))

    optimizer = getattr(optim, args.optimizer)  
    optimizer = optimizer(model.parameters(), lr=lr, weight_decay=args.weight_decay, betas=betas)
    
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    model, optimizer = accelerator.prepare(model, optimizer)
    
    creterion = torch.nn.MSELoss() # hybrid_loss with chamfer_loss # reduction='none'
    global_step = 0
    for epoch in range(num_epochs):

        model.train()
        total_loss = 0
        tqdm_train_loader = tqdm(dataloader_train, desc=f"Epoch {epoch + 1}/{num_epochs} Training")
        for pca_recon, pca_rep, pca_input in tqdm_train_loader:

            pca_residual= pca_input - pca_recon


            # Apply wavelet transform to PCA reconstruction
            recon_coeffs = wavelet_transform_tensor(pca_recon, wavelet='db4', level=3)

            print("Recon coeffs shape:", recon_coeffs.shape)
            
            # Apply wavelet transform to PCA residual
            residual_coeffs = wavelet_transform_tensor(pca_residual, wavelet='db4', level=3)


            print("Residual coeffs shape:", residual_coeffs.shape)

            exit()


            def analyze_frequency_components(coeffs_batch):
                """
                Analyze frequency components from wavelet coefficients.
                Returns statistical measures for each level.
                """
                stats = []
                for b in range(len(coeffs_batch)):
                    batch_stats = []
                    for c in range(len(coeffs_batch[0])):
                        level_stats = []
                        coeffs = coeffs_batch[b][c]
                        
                        # Calculate statistics for each decomposition level
                        for level_coeffs in coeffs[1:]:  # Skip the approximation coefficients
                            h, v, d = level_coeffs
                            level_energy = np.sum(h**2) + np.sum(v**2) + np.sum(d**2)
                            level_stats.append({
                                'energy': level_energy,
                                'mean_magnitude': (np.mean(np.abs(h)) + np.mean(np.abs(v)) + np.mean(np.abs(d))) / 3,
                                'max_magnitude': max(np.max(np.abs(h)), np.max(np.abs(v)), np.max(np.abs(d)))
                            })
                        batch_stats.append(level_stats)
                    stats.append(batch_stats)
                return stats
            
            # Analyze frequency components
            recon_freq_stats = analyze_frequency_components(recon_coeffs)
            residual_freq_stats = analyze_frequency_components(residual_coeffs)
            
            # Optional: Reconstruct signals if needed
            reconstructed_recon = inverse_wavelet_transform_tensor(recon_coeffs, 
                                                                wavelet='db4', 
                                                                original_shape=pca_recon.shape)
            reconstructed_residual = inverse_wavelet_transform_tensor(residual_coeffs, 
                                                                    wavelet='db4', 
                                                                    original_shape=pca_residual.shape)
            
            # Verify reconstruction accuracy
            recon_error = torch.mean((pca_recon - reconstructed_recon)**2)
            residual_error = torch.mean((pca_residual - reconstructed_residual)**2)

            print(f"Reconstruction Error: {recon_error.item()}")
            print(f"Residual Error: {residual_error.item()}")
    


            # freuency domain of the pca_recon

            # freuency domain of the pca_residual


            exit()


            
            
            
            # deformed_points = model(pca_recon, pca_rep)
            # loss = 100.0*creterion(deformed_points, pca_input)# shape of loss: torch.Size([2, 1024, 3]) 

            # print("val of loss:", loss) #  0.0023---> real loss:0.000023

            # loss= mean_flat(loss) / args.batch_size
            # print("shape of loss:", loss.shape) # shape of loss: torch.Size([2])
            # exit()

            # optimizer.zero_grad()
            # # loss.backward()
            # accelerator.backward(loss)
            # clip_grad_norm_(model.parameters(), max_norm=1.0)
            # optimizer.step()

            # total_loss += loss.item()

            
            # wandb.log({"train_loss": loss.item()}, step=global_step)
            tqdm_train_loader.set_postfix(loss=f"{loss.item():.4f}")
            

            # print("shape of pca_recon:", pca_recon.shape)
            # print("shape of pca_rep:", pca_rep.shape)
            # print("shape of pca_input:", pca_input.shape)
            # shape of pca_recon: torch.Size([8, 1024, 3]) 
            # shape of pca_rep: torch.Size([8, 1, 64])
            # shape of pca_input: torch.Size([8, 1024, 3])
            # print("device of pca_recon:", pca_recon.device) # cuda:0
            # print("device of pca_rep:", pca_rep.device)
            # print("device of pca_input:", pca_input.device)

        

        model.eval()
        with torch.no_grad():
            val_loss = 0
            num_batches= len(dataloader_val)
            # print("num_batches:", num_batches)# 86
            for pca_recon, pca_rep, pca_input in dataloader_val:

                loss = creterion(model(pca_recon, pca_rep), pca_input)
                val_loss += loss.item()
                # print("shape of pca_recon:", pca_recon.shape)
                # print("shape of pca_rep:", pca_rep.shape)
                # print("shape of pca_input:", pca_input.shape)
                # shape of pca_recon: torch.Size([8, 1024, 3])
                # shape of pca_rep: torch.Size([8, 1, 64])
                # shape of pca_input: torch.Size([8, 1024, 3])


            # val_loss /= num_batches

        # wandb.log({
        #     "val_loss": val_loss,
        # }, step=global_step)


        global_step += 1


        # Inside your training loop, after the validation step
        # if epoch % 10 == 0:  # Log every 10 epochs
        #     # Get a batch of validation data to visualize
        #     val_batch = next(iter(dataloader_val))
        #     pca_recon_val, pca_rep_val, pca_input_val = val_batch
            
        #     with torch.no_grad():
        #         deformed_points_val = model(pca_recon_val, pca_rep_val)                
        #         # Convert to numpy and take first 8 examples
        #         deformed_points_np = deformed_points_val.cpu().numpy()[:8]  # shape: (8, 1024, 3)
        #         target_points_np = pca_input_val.cpu().numpy()[:8]         # shape: (8, 1024, 3)
                
        #         # Create a dictionary to store all point clouds
        #         point_clouds_dict = {}
                
        #         # Log all 8 shapes for both deformed and target point clouds
        #         for i in range(8):
        #             point_clouds_dict[f"deformed_shape_{i}_epoch_{epoch}"] = wandb.Object3D(deformed_points_np[i])
        #             point_clouds_dict[f"target_shape_{i}_epoch_{epoch}"] = wandb.Object3D(target_points_np[i])
                
        #         # Log all point clouds at once
        #         wandb.log(point_clouds_dict)


        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), 'best_model_weights.pth')
        #     wandb.log({"best_val_loss": best_val_loss})

        # torch.save(model.state_dict(), 'latest_model_weights.pth')

            


        




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
    train(args)


    #test() visualize the result

if __name__ == '__main__':
    main()