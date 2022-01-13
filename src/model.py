import os

import torch
import pytorch_lightning as pl

from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM 
from transformers import get_scheduler


optim_dict = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW
}

def prepare_optimizer(params, optimizer_hparams):
    name = optimizer_hparams['name']
    hparams = {k:v for k,v in optimizer_hparams.items() if k != 'name'}
    return optim_dict[name](params, **hparams)


def prepare_scheduler(optimizer, scheduler_hparams):
    num_training_steps = scheduler_hparams['num_train_steps']
    num_warmup_steps = int(num_training_steps * scheduler_hparams['warmup_ratio'])
    scheduler = get_scheduler(scheduler_hparams['name'], optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    return scheduler


def get_param_groups(model, weight_decay):
    no_decay = ["bias", "bn", "ln", "norm"]
    param_groups = [
        {
            # apply weight decay
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": weight_decay
        },
        {
            # not apply weight decay
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return param_groups

class BaseModel(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.teacher = self.prepare_teacher()
        self.student = self.prepare_student()
        self.tokenizer = self.prepare_tokenizer()

    def prepare_teacher(self):
        teacher = AutoModel.from_pretrained(**self.hparams.teacher)
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher
    
    def prepare_student(self):
        config = AutoConfig.from_pretrained(**self.hparams.student)
        student = AutoModel.from_config(config)
        return student
    
    def prepare_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.hparams.teacher.pretrained_model_name_or_path)
        return tokenizer

    def configure_optimizers(self):
        student_param_groups = get_param_groups(self.student, self.hparams.optimizer.weight_decay)
        optimizer = prepare_optimizer(student_param_groups, self.hparams.optimizer)
        scheduler = prepare_scheduler(optimizer, self.hparams.scheduler)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        ckpt_dir = os.path.join(self.hparams.ckpt_dir, f'{self.trainer.global_step + 1:06d}')
        self.student.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
    
    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')



