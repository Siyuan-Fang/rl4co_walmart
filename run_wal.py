import torch
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary, LearningRateMonitor, EarlyStopping, Callback
# from lightning.pytorch.cli import LRSchedulerConfig

from rl4co.envs.graph.wal.env import WALEnv
from rl4co.models.zoo.am import AttentionModel
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.loggers import WandbLogger
from swanlab.integration.pytorch_lightning import SwanLabLogger
import pandas as pd
# import wandb


class GradientMonitor(Callback):

    
    def on_after_backward(self, trainer, pl_module):

        total_norm = 0
        max_grad = 0
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, param_norm.item())
        
        total_norm = total_norm ** (1. / 2)
        

        if trainer.logger is not None:
            trainer.logger.log_metrics({
                "grad/total_norm": total_norm,
                "grad/max_grad": max_grad
            }, step=trainer.global_step)
        

        if total_norm > 10.0:  
            print(f"⚠️ Warning: Large gradient norm detected: {total_norm:.4f}")
        if max_grad > 100.0:  
            print(f"⚠️ Warning: Large individual gradient detected: {max_grad:.4f}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = WALEnv(generator_params=dict(
    num_loc=730,  # Reduced from 50 for MPS compatibility  
    to_choose=90,  # Reduced from 10 for MPS compatibility
    loc_sampler=True,
    memory_efficient="ultra",  # Use ultra-efficient mode
    device=device.type  # Force GPU-native generation to avoid CPU->GPU transfers
))

# Ensure environment is on the correct device
env = env.to(device)

# Test dynamic feature computation
print("Testing dynamic WAL feature computation...")


# Model: default is AM with REINFORCE and greedy rollout baseline
td_init = env.reset(batch_size=[3])
model = AttentionModel(env,
                       baseline='rollout',
                       train_data_size=100_000, # Reduced size for testing
                       val_data_size=10_000,   # Reduced size for testing
                       optimizer_kwargs={'lr': 2e-5},  
                       lr_scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,  # use ReduceLROnPlateau
                       lr_scheduler_kwargs={
                           'mode': 'max',  
                           'factor': 0.3,
                           'patience': 3, 
                           'verbose': True, 
                           'min_lr': 1e-7, 
                           'threshold': 0.01, 
                           'cooldown': 2 
                       },
                       lr_scheduler_interval='epoch',  # update every epoch
                       lr_scheduler_monitor='val/reward'  # monitor validation reward
                       )

# Move model to device to ensure proper device management
model = model.to(device)
print(f"✅ Model moved to device: {device}")

# Ensure environment is synchronized with model device
env = env.to(device)
print(f"✅ Environment synchronized to device: {device}")

## Test greedy rollout with untrained model and plot
# Greedy rollouts over untrained policy
print("\n🧪 Testing policy...")
try:
    policy = model.policy.to(device)
    print("✅ Policy moved to device")
    
    print("Testing policy forward pass...")
    out = policy(td_init.clone(), env, phase="test", decode_type="greedy")#TODO：尝试在这里能搞好
    print(f"✅ Policy test successful: {out['reward'].shape}")
    
except Exception as e:
    print(f"❌ Error during policy test: {e}")
    import traceback
    traceback.print_exc()

#==============================================training==============================================
## Callbacks
# Checkpointing callback: save models when validation reward improves
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints", # save to checkpoints/
    filename="epoch_{epoch:03d}",  # save as epoch_XXX.ckpt
    save_top_k=1, # save only the best model
    save_last=True, # save the last model
    monitor="val/reward", # monitor validation reward
    mode="max") # maximize validation reward

# Print model summary
rich_model_summary = RichModelSummary(max_depth=10)

# Learning rate monitor
learning_rate_monitor = LearningRateMonitor(logging_interval="step")

# gradient monitor
gradient_monitor = GradientMonitor()

# Callbacks list
callbacks = [checkpoint_callback, rich_model_summary, learning_rate_monitor, gradient_monitor]

logger = SwanLabLogger(project="rl4co",name="wal-am-dynamic")
## Trainer
trainer = RL4COTrainer(
    max_epochs=10,  # Reduced for testing dynamic computation
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=callbacks,
    gradient_clip_val=0.05,  
    gradient_clip_algorithm="norm",  # Use norm-based clipping
    enable_progress_bar=True,
    enable_model_summary=True,
    enable_checkpointing=True,
    precision="16-mixed",  # use mixed precision training for numerical stability
)
## fit the model
trainer.fit(model)
