# The MIT License (MIT)
# © 2024 templar.tech

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
# fmt: off

# Standard library
import sys
import time
import random
import asyncio
import argparse
import threading
import os

# Third party
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
import bittensor as bt
from torch.optim import SGD
from transformers import LlamaForCausalLM
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.nn.parallel import DistributedDataParallel as DDP

# Local
import tplr


# GPU optimizations
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Miner:
    
    # Command line config items.
    @staticmethod
    def config():
        parser = argparse.ArgumentParser(description='Miner script')
        parser.add_argument('--netuid', type=int, default=268, help='Bittensor network UID.')
        parser.add_argument('--project', type=str, default='templar', help='Wandb project.')
        parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        parser.add_argument('--trace', action='store_true', help='Enable trace logging')
        parser.add_argument('--store-gathers', action='store_true', help='Store gathered gradients in R2')
        parser.add_argument('--world-size', type=int, default=torch.cuda.device_count())
        parser.add_argument('--port', type=int, default=29500)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        config = bt.config(parser)
        if config.debug:
            tplr.debug()
        if config.trace:
            tplr.trace()
        return config
    
    def __init__(self, config, local_rank):
        self.local_rank = local_rank
        self.device = f'cuda:{local_rank}'
        self.is_rank0 = local_rank == 0
        
        # Init config and load hparams
        self.config = config
        self.hparams = tplr.load_hparams()
        
        # Init bittensor objects (only on rank 0)
        if self.is_rank0:
            self.wallet = bt.wallet(config=self.config)
            self.subtensor = bt.subtensor(config=self.config)
            self.metagraph = self.subtensor.metagraph(self.config.netuid)
            if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys:
                tplr.logger.error(f'\n\t[bold]The wallet {self.wallet} is not registered on subnet: {self.metagraph.netuid}[/bold]')
                sys.exit()
            self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            uid_tensor = torch.tensor([self.uid], device=self.device)
        else:
            uid_tensor = torch.tensor([0], device=self.device)
        
        # Broadcast uid to all ranks
        dist.broadcast(uid_tensor, 0)
        self.uid = uid_tensor.item()
        
        # Init model with DDP
        self.model = LlamaForCausalLM(self.hparams.model_config)
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[local_rank])
        self.tokenizer = self.hparams.tokenizer
        
        # Init optimizer and momentum
        self.optimizer = SGD(self.model.parameters(), lr=self.hparams.learning_rate)
        self.momentum = {n: torch.zeros_like(p) for n, p in self.model.module.named_parameters()}
        
        # Initialize schedulers (same as before)
        self.scheduler = self._init_scheduler()
        
        # Init compression
        self.transformer = tplr.compress.TransformDCT(self.model.module, target_chunk=self.hparams.target_chunk)
        self.compressor = tplr.compress.CompressDCT()
        
        # Init comms (only on rank 0)
        if self.is_rank0:
            self.comms = tplr.comms.Comms(
                wallet=self.wallet,
                config=self.config,
                netuid=self.config.netuid,
                metagraph=self.metagraph,
                hparams=self.hparams,
                uid=self.uid,
                device=self.device
            )
            self.bucket = self.comms.get_own_bucket('gradients', 'read')
            self.comms.try_commit(self.wallet, self.bucket)
            self.comms.fetch_commitments()

        # Init state params
        self.stop_event = asyncio.Event()
        self.current_block = self.subtensor.block
        self.current_window = int(self.current_block / self.hparams.blocks_per_window)
        self.start_window = self.current_window  # Record the start window
        self.global_step = 0  # Initialize global_step to zero
        self.comms.current_window = self.current_window 
        self.step_counter = 0

        # Add step tracking
        self.window_step = 0
        
        # Track additional metrics
        self.total_tokens_processed = 0
        self.batch_times = []  # For tracking processing speed
        
        # Initialize WandB
        self.wandb = tplr.initialize_wandb(
            run_prefix='M',
            uid=self.uid,
            config=self.config,
            group='miner',
            job_type='mining'
        )

    # Main training loop.
    async def run(self):
        # Initialize training state
        if self.is_rank0:
            self.start_window = await self.comms.get_start_window()
            window_tensor = torch.tensor([self.start_window], device=self.device)
        else:
            window_tensor = torch.tensor([0], device=self.device)
        
        # Broadcast checkpoint status and data
        dist.broadcast(window_tensor, 0)
        self.start_window = window_tensor.item()
        
        # Load checkpoint (only rank 0)
        if self.is_rank0:
            success, momentum, global_step, optimizer, scheduler = await self.comms.load_checkpoint(
                model=self.model.module,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                transformer=self.transformer,
                compressor=self.compressor,
                current_window=self.current_window,
                device=self.device,
                peers=self.peers,
                uid=self.uid
            )
            success_tensor = torch.tensor([success], device=self.device)
        else:
            success_tensor = torch.tensor([False], device=self.device)
            
        # Broadcast checkpoint status and data
        dist.broadcast(success_tensor, 0)
        if success_tensor.item():
            self._broadcast_checkpoint_data(momentum, global_step)

        # Training loop
        while True:
            step_window = self.current_window
            
            # Get sharded pages for this window
            pages = await self._get_ddp_pages(step_window)
            
            # Training logic
            total_loss = 0
            for batch in self._get_dataloader(pages):
                loss = self._training_step(batch)
                total_loss += loss
                
                # Only rank 0 handles peer communication
                if self.is_rank0:
                    gradient, xshapes, totalks = self._compress_gradients()
                    gather_result = await self.comms.gather(
                        state_dict=gradient,
                        my_uid=self.uid,
                        uids=self.peers,
                        window=step_window,
                        key='gradient',
                        timeout=30,
                        device=self.device,
                        local=False,
                        stale_retention=100,
                        global_step=self.global_step,
                        store_gathers=self.config.store_gathers
                    )
                else:
                    gather_result = None
                
                # Broadcast and apply gathered gradients
                gather_result = self._broadcast_from_rank0(gather_result)
                if gather_result is not None:
                    self._apply_gathered_gradients(gather_result)
                
                self.optimizer.step()
                self.scheduler.step()
            
            # Checkpointing (only rank 0)
            if self.is_rank0 and self.global_step % self.hparams.checkpoint_frequency == 0:
                await self.comms.save_checkpoint(
                    model=self.model.module,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    momentum=self.momentum,
                    global_step=self.global_step,
                    current_window=self.current_window,
                    start_window=self.start_window
                )

    # Listens for new blocks and sets self.current_block and self.current_window
    def block_listener(self, loop):
        def handler(event, _u, _s):
            self.current_block = int(event['header']['number']) #type: ignore
            new_window = int(self.current_block / self.hparams.blocks_per_window)
            if new_window != self.current_window:
                self.current_window = new_window
                self.comms.current_window = self.current_window  # Synchronize comms current_window
        while not self.stop_event.is_set():
            try:
                bt.subtensor(config=self.config).substrate.subscribe_block_headers(handler)
                break
            except Exception:
                time.sleep(1)

    # Helper methods
    async def _get_ddp_pages(self, step_window):
        """Get sharded pages for DDP training"""
        if self.is_rank0:
            total_pages = await tplr.r2_dataset.R2DatasetLoader.next_pages(
                offset=step_window,
                n_pages=self.hparams.pages_per_window * dist.get_world_size(),
                seed=self.uid
            )
        else:
            total_pages = None
        
        total_pages = self._broadcast_from_rank0(total_pages)
        pages_per_rank = len(total_pages) // dist.get_world_size()
        start_idx = self.local_rank * pages_per_rank
        end_idx = start_idx + pages_per_rank
        return total_pages[start_idx:end_idx]

# Start miner/validator.
if __name__ == "__main__":
    asyncio.run(Miner().run())

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
