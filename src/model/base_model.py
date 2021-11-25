import os

import torch
import pytorch_lightning as pl

from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers import get_scheduler


class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.teacher, self.student, self.tokenizer = self.prepare_models()
    
    def prepare_models(self):
        teacher = AutoModel.from_pretrained(
            self.hparams.teacher.name_or_model_path,
            output_attentions = True,
            output_hidden_states = True
        )
        
        config = AutoConfig.from_pretrained(
            self.hparams.teacher.name_or_model_path,
            output_attention = True,
            output_hidden_states = True,
            **self.hparams.student
        )
        
        student = AutoModel.from_config(config)
        
        for param in teacher.parameters():
            param.requires_grad = False
        
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.teacher.name_or_model_path)
        return teacher, student, tokenizer
    
    
    def student_param_groups(self):
        no_decay = ["bias", "bn", "ln", "norm"]
        param_groups = [
            {
                # apply weight decay
                "params": [p for n, p in self.student.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
                "weight_decay": self.hparams.optim.weight_decay
            },
            {
                # not apply weight decay
                "params": [p for n, p in self.student.named_parameters() if any(nd in n.lower() for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return param_groups


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.student_param_groups(), 
            lr = self.hparams.optim.lr, 
            betas = self.hparams.optim.betas,
            weight_decay = self.hparams.optim.weight_decay,
            eps = self.hparams.optim.adam_epsilon,
        )

        num_training_steps = self.hparams.optim.max_steps
        num_warmup_steps = int(num_training_steps * self.hparams.optim.warmup_ratio)
        scheduler = get_scheduler(self.hparams.optim.scheduler, optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': self.hparams.optim.accumulate_grad_batches,
            }
        }


    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        save_dir = os.path.join(self.hparams.ckpt_dir, 'transformers')
        self.student.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)