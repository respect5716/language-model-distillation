import torch
import torch.nn as nn
import torch.nn.function as F

from .base_model import BaseModel

class Model(BaseModel):
    def step(self, batch, phase):
        to = self.student(**batch)
        so = self.teacher(**batch)

        th = to.hidden_states
        sh = so.hidden_states

        te = relation_attention(th[0], th[0], self.hparams.num_relation_heads, batch.attention_mask) # teacher embedding attn
        se = relation_attention(sh[0], sh[0], self.hparams.num_relation_heads, batch.attention_mask) # student embedding attn

        tl = relation_attention(th[-1], th[-1], self.hparams.num_relation_heads, batch.attention_mask) # teacher last hidden state attn
        sl = relation_attention(sh[-1], sh[-1], self.hparams.num_relation_heads, batch.attention_mask) # student last hidden state attn

        tel = relation_attention(th[0], th[-1], self.hparams.num_relation_heads, batch.attention_mask) # teacher embedding -> last hidden attn
        sel = relation_attention(sh[0], sh[-1], self.hparams.num_relation_heads, batch.attention_mask) # student embedding -> last hidden attn

        tle = relation_attention(th[-1], th[0], self.hparams.num_relation_heads, batch.attention_mask) # teacher last hidden -> embedding attn
        sle = relation_attention(sh[-1], sh[0], self.hparams.num_relation_heads, batch.attention_mask) # student last hidden -> embedding attn

        loss_e = kl_div_loss(se, te, temperature=self.hparams.temperature)
        loss_l = kl_div_loss(sl, tl, temperature=self.hparams.temperature)
        loss_el = kl_div_loss(sel, tel, temperature=self.hparams.temperature)
        loss_le = kl_div_loss(sle, tle, temperature=self.hparams.temperature)
        loss = loss_e + loss_l + loss_el + loss_le

        log = {f'{phase}/loss': loss, f'{phase}/loss_e': loss_e, f'{phase}/loss_l': loss_l, f'{phase}/loss_el': loss_el, f'{phase}/loss_le': loss_le}
        self.log_dict({f'{phase}/loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')


def transpose_for_scores(h, num_heads):
    batch_size, seq_length, dim = h.size()
    head_size = dim // num_heads
    h = h.view(batch_size, seq_length, num_heads, head_size)
    return h.permute(0, 2, 1, 3)


def relation_attention(h1, h2, num_relation_heads, attention_mask=None):        
    h1 = transpose_for_scores(h1, num_relation_heads) # (batch, num_heads, seq_length, head_size)
    h2 = transpose_for_scores(h2, num_relation_heads) # (batch, num_heads, seq_length, head_size)

    attn = torch.matmul(h1, h2.transpose(-1, -2)) # (batch_size, num_heads, seq_length, seq_length)
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1 - attention_mask) * -10000.0
        attn = attn + attention_mask

    attn = attn.view(-1, attn.size(-1)) # (~, seq_length)
    return attn

def kl_div_loss(s, t, temperature=1.):
    s = F.log_softmax(s / temperature, dim=-1)
    t = F.softmax(t / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean')