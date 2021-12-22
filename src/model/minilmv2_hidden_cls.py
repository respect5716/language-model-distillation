import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .base_model import BaseModel

class Model(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

    def prepare(self):
        teacher, student, tokenizer = super().prepare()
        teacher = to_distill(teacher)
        student = to_distill(student)
        return teacher, student, tokenizer


    def step(self, batch, phase):
        teacher_outputs = self.teacher(**batch)
        student_outputs = self.student(**batch)

        # cls loss
        teacher_cls = teacher_outputs.last_hidden_state[:, 0].unsqueeze(0) # (1, bs, dim)
        student_cls = student_outputs.last_hidden_state[:, 0].unsqueeze(0) # (1, bs, dim)
        # loss_cls = distance_loss_fn(student_cls, teacher_cls) * 10
        loss_cls = minilm_loss(teacher_cls, student_cls, self.hparams.num_relation_heads, None, self.hparams.temperature)

        # minilm loss
        teacher_qkv = get_qkvs(self.teacher)[self.hparams.teacher_layer_index] # (batch, head, seq, head_dim)
        student_qkv = get_qkvs(self.student)[self.hparams.student_layer_index] # (batch, head, seq, head_dim)

        teacher_hidden = teacher_outputs.hidden_states[self.hparams.teacher_layer_index]
        student_hidden = student_outputs.hidden_states[self.hparams.student_layer_index]

        loss_q = minilm_loss(teacher_qkv['q'], student_qkv['q'], self.hparams.num_relation_heads, batch.attention_mask, self.hparams.temperature)
        loss_k = minilm_loss(teacher_qkv['k'], student_qkv['k'], self.hparams.num_relation_heads, batch.attention_mask, self.hparams.temperature)
        loss_v = minilm_loss(teacher_qkv['v'], student_qkv['v'], self.hparams.num_relation_heads, batch.attention_mask, self.hparams.temperature)
        loss_hidden = minilm_loss(teacher_hidden, student_hidden, self.hparams.num_relation_heads, batch.attention_mask, self.hparams.temperature)        
        
        loss = loss_cls + loss_q + loss_k + loss_v + loss_hidden

        log = {f'{phase}/loss': loss, f'{phase}/loss_cls': loss_cls, f'{phase}/loss_q': loss_q, f'{phase}/loss_k': loss_k, f'{phase}/loss_v': loss_v, f'{phase}/loss_hidden': loss_hidden}
        self.log_dict(log, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')


def to_distill(model):
    model.base_model.encoder.layer[0].attention.self.__class__._forward = bert_self_attention_forward
    model.base_model.encoder.layer[0].attention.output.__class__._forward = bert_self_output_forward
    model.base_model.encoder.layer[0].intermediate.__class__._forward = bert_intermediate_forward
    model.base_model.encoder.layer[0].output.__class__._forward = bert_output_forward

    for layer in model.base_model.encoder.layer:
        layer.attention.self.forward = layer.attention.self._forward
        layer.attention.output.forward = layer.attention.output._forward
        layer.intermediate.forward = layer.intermediate._forward
        layer.output.forward = layer.output._forward

    return model


def get_qkvs(model):
    attns = [l.attention.self for l in model.base_model.encoder.layer]
    qkvs = [{'q': a.q, 'k': a.k, 'v': a.v} for a in attns]    
    return qkvs


def get_hiddens(model):
    hiddens = []
    for layer in model.base_model.encoder.layer:
        hidden = {}
        hidden['hidden1'] = layer.attention.output.hidden
        hidden['hidden2'] = layer.intermediate.hidden
        hidden['hidden3'] = layer.output.hidden
        hiddens.append(hidden)
    return hiddens


def transpose_for_scores(h, num_heads):
    batch_size, seq_length, dim = h.size()
    head_size = dim // num_heads
    h = h.view(batch_size, seq_length, num_heads, head_size)
    return h.permute(0, 2, 1, 3) # (batch, num_heads, seq_length, head_size)


def attention(h1, h2, num_heads, attention_mask=None):
    assert h1.size() == h2.size()
    head_size = h1.size(-1) // num_heads
    h1 = transpose_for_scores(h1, num_heads) # (batch, num_heads, seq_length, head_size)
    h2 = transpose_for_scores(h2, num_heads) # (batch, num_heads, seq_length, head_size)

    attn = torch.matmul(h1, h2.transpose(-1, -2)) # (batch_size, num_heads, seq_length, seq_length)
    attn = attn / math.sqrt(head_size)
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1 - attention_mask) * -10000.0
        attn = attn + attention_mask

    return attn


def kl_div_loss(s, t, temperature):
    if len(s.size()) != 2:
        s = s.view(-1, s.size(-1))
        t = t.view(-1, t.size(-1))

    s = F.log_softmax(s / temperature, dim=-1)
    t = F.softmax(t / temperature, dim=-1)
    return F.kl_div(s, t, reduction='batchmean')


def minilm_loss(t, s, num_relation_heads, attention_mask=None, temperature=1.0):
    attn_t = attention(t, t, num_relation_heads, attention_mask)
    attn_s = attention(s, s, num_relation_heads, attention_mask)
    loss = kl_div_loss(attn_s, attn_t, temperature=temperature)
    return loss


def cdist(v):
    d = torch.cdist(v, v, p=2)
    m = d[d>0].mean()
    return d / m

def distance_loss_fn(s, t):
    s = s.contiguous()
    t = t.contiguous()
    td = cdist(t)
    sd = cdist(s)
    return F.smooth_l1_loss(sd, td, reduction='mean')


def bert_self_attention_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    past_key_value=None,
    output_attentions=False,
):
    mixed_query_layer = self.query(hidden_states)
    mixed_key_layer = self.key(hidden_states)
    mixed_value_layer = self.value(hidden_states)
    
    query_layer = self.transpose_for_scores(mixed_query_layer)
    key_layer = self.transpose_for_scores(mixed_key_layer)
    value_layer = self.transpose_for_scores(mixed_value_layer)
    
    self.q = mixed_query_layer # (Batch, Seq, Dim)
    self.k = mixed_key_layer # (Batch, Seq, Dim)
    self.v = mixed_value_layer # (Batch, Seq, Dim)

    if self.is_decoder:
        past_key_value = (key_layer, value_layer)

    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
        seq_length = hidden_states.size()[1]
        position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
        position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
        distance = position_ids_l - position_ids_r
        positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
        positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

        if self.position_embedding_type == "relative_key":
            relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores
        elif self.position_embedding_type == "relative_key_query":
            relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
            relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
            attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = nn.Softmax(dim=-1)(attention_scores)
    attention_probs = self.dropout(attention_probs)

    if head_mask is not None:
        attention_probs = attention_probs * head_mask

    context_layer = torch.matmul(attention_probs, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
    if self.is_decoder:
        outputs = outputs + (past_key_value,)
    return outputs


def bert_self_output_forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    self.hidden = hidden_states
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states


def bert_intermediate_forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    self.hidden = hidden_states
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states


def bert_output_forward(self, hidden_states, input_tensor):
    hidden_states = self.dense(hidden_states)
    self.hidden = hidden_states
    hidden_states = self.dropout(hidden_states)
    hidden_states = self.LayerNorm(hidden_states + input_tensor)
    return hidden_states
