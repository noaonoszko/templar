import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from argparse import ArgumentParser

from neurons.miner import Miner

def setup_ddp(rank, world_size, port):
    """Initialize DDP process group"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

async def run_miner(rank, world_size, port, args):
    setup_ddp(rank, world_size, port)
    miner = Miner(args, rank)
    await miner.run()
    dist.destroy_process_group()

def main():
    parser = Miner.config()
    parser.add_argument('--world-size', type=int, default=torch.cuda.device_count())
    parser.add_argument('--port', type=int, default=29500)
    args = parser.parse_args()

    if args.world_size > torch.cuda.device_count():
        print(f"Requested {args.world_size} GPUs but only {torch.cuda.device_count()} available")
        sys.exit(1)

    mp.spawn(
        run_miner,
        args=(args.world_size, args.port, args),
        nprocs=args.world_size,
        join=True
    )

if __name__ == "__main__":
    main()