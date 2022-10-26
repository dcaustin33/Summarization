import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import time
import os

torch.backends.cudnn.enabled = True 
torch.backends.cudnn.benchmark = True
    
class Evaluator:
    def __init__(self, 
                 model: nn.Module,
                 val_dataloader: DataLoader,
                 args, 
                 validation_step,
                 dataset,
                 val_metrics: dict = None,
                 wandb = None, 
                 file_name = None):
        
        self.model = model
        self.dataset = dataset
        self.val_dataloader = val_dataloader
        self.args = args
        self.validation_step = validation_step
        self.val_metrics = val_metrics
        self.wandb = wandb
        self.file_name = file_name
        
        
    def evaluate(self):
        print('Now Evaluating')
        print('Evaluating for', self.args.val_steps, 'steps')
        now = time.time()

        self.model = self.model.cuda()
        scaler = torch.cuda.amp.GradScaler()
        
        val_steps = 0
        with torch.no_grad():
            while val_steps < self.args.val_steps:
                for k, val_data in enumerate(self.val_dataloader):
                    val_data['article'] = self.dataset.tokenizer(val_data['article_text'], max_length=self.dataset.max_length, truncation=True, padding='longest', return_tensors="pt")
                    val_data['summary'] = self.dataset.tokenizer(val_data['summary_text'], max_length=self.dataset.max_length, truncation=True, padding='longest', return_tensors="pt")
                    if val_steps % self.args.log_n_val_steps == 0 and val_steps != 0:
                        loss = self.validation_step(val_data, self.model, self.val_metrics, val_steps, log = True, wandb = self.wandb, args = self.args, file_name = self.file_name)
                        break
                    else:
                        loss = self.validation_step(val_data, self.model, self.val_metrics, val_steps, log = False, wandb = self.wandb, args = self.args, file_name = self.file_name)
                    
                    val_steps += 1
