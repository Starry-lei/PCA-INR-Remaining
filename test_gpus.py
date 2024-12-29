import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
def main():
    # Basic setup without any initial barriers
    accelerator = Accelerator()
    local_rank = accelerator.local_process_index


    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")


    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    
    # Print initial info
    print(f"Process {os.getpid()} with local_rank {local_rank} starting")
    
    try:
        # Create a very simple model
        model = nn.Linear(2, 2)
        print(f"Rank {local_rank}: Created model")
        
        # Create optimizer
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        print(f"Rank {local_rank}: Created optimizer--------")


        optimizer = accelerator.prepare(optimizer)
        print(f"Rank {local_rank}: Completed prepare optimizer?")


        print(f"Rank {local_rank}: About to move model to {accelerator.device}")
        try:
            model = model.to(accelerator.device)
            print(f"Rank {local_rank}: Successfully moved model to {next(model.parameters()).device}")
        except Exception as e:
            print(f"Rank {local_rank}: Failed to move model to device: {str(e)}")


        accelerator.wait_for_everyone()
        
        try:
            model = accelerator.prepare(model)
            print(f"Rank {local_rank}: Successfully prepared model")
        except Exception as e:
            print(f"Rank {local_rank}: Failed to prepare model: {str(e)}")



        # # Wrap model in DDP directly
        # print(f"Rank {local_rank}: About to wrap model in DDP")
        # model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        # print(f"Rank {local_rank}: Successfully wrapped model in DDP")


        # Now try preparing the model separately
        print(f"Rank {local_rank}: About to prepare model")


        print(f"Rank {local_rank}: Completed prepare model!!!!!!!!!!!!")
        
        
        # Do a simple forward pass
        dummy_input = torch.randn(4, 2)
        dummy_input = accelerator.prepare(dummy_input)
        output = model(dummy_input)
        print(f"Rank {local_rank}: Completed forward pass")
        
        # Force a sync point
        accelerator.wait_for_everyone()
        print(f"Rank {local_rank}: Passed sync point")
        
    except Exception as e:
        print(f"Rank {local_rank}: Error occurred: {str(e)}")
    finally:
        # Clean shutdown
        if torch.distributed.is_initialized():
            try:
                accelerator.wait_for_everyone()
                print(f"Rank {local_rank}: Starting cleanup")
                torch.distributed.destroy_process_group()
                print(f"Rank {local_rank}: Cleanup complete")
            except Exception as e:
                print(f"Rank {local_rank}: Cleanup failed: {str(e)}")

if __name__ == "__main__":
    main()