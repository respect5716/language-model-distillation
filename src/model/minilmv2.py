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
        teacher_outputs = self.teacher(input_ids=batch.input_ids, attention_mask=batch.attention_mask)
        student_outputs = self.student(input_ids=batch.input_ids, attention_mask=batch.attention_mask)

        teacher_qkv = get_qkvs(self.teacher)[self.hparams.teacher_layer_index] # (batch, head, seq, head_dim)
        student_qkv = get_qkvs(self.student)[self.hparams.student_layer_index] # (batch, head, seq, head_dim)

        loss_q = minilm_loss(teacher_qkv['q'], student_qkv['q'], num_relation_heads=self.hparams.num_relation_heads, attention_mask=batch.attention_mask)
        loss_k = minilm_loss(teacher_qkv['k'], student_qkv['k'], num_relation_heads=self.hparams.num_relation_heads, attention_mask=batch.attention_mask)
        loss_v = minilm_loss(teacher_qkv['v'], student_qkv['v'], num_relation_heads=self.hparams.num_relation_heads, attention_mask=batch.attention_mask)
        loss = loss_q + loss_k + loss_v

        log = {f'{phase}/loss': loss, f'{phase}/loss_q': loss_q, f'{phase}/loss_k': loss_k, f'{phase}/loss_v': loss_v}
        self.log_dict({f'{phase}/loss': loss}, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'valid')

    def test_step(self, batch, batch_idx):
        return self.step(batch, 'test')


def to_distill(model):
    model.base_model.encoder.layer[0].attention.self.__class__._forward = bert_self_attention_forward
    for layer in model.base_model.encoder.layer:
        layer.attention.self.forward = layer.attention.self._forward
    
    return model


def get_qkvs(model):
    attns = [l.attention.self for l in model.base_model.encoder.layer]
    qkvs = [{'q': a.q, 'k': a.k, 'v': a.v} for a in attns]    
    return qkvs


def relation_attention(h, num_relation_heads, attention_mask=None):        
    batch_size, seq_length, dim = h.size()
    relation_head_size = dim // num_relation_heads

    h = h.view(batch_size, seq_length, num_relation_heads, relation_head_size)
    h = h.permute(0, 2, 1, 3) # (batch_size, num_relation_heads, seq_length, attention_head_size)

    attn = torch.matmul(h, h.transpose(-1, -2)) # (batch_size, num_relation_heads, seq_length, seq_length)
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1 - attention_mask) * -10000.0
        attn = attn + attention_mask

    attn = attn.view(-1, seq_length)
    return attn


def minilm_loss(t, s, num_relation_heads, attention_mask=None):
    attn_t = relation_attention(t, num_relation_heads, attention_mask)
    attn_s = relation_attention(s, num_relation_heads, attention_mask)
    loss = F.kl_div(F.log_softmax(attn_s, dim=-1), F.softmax(attn_t, dim=-1), reduction='batchmean')
    return loss


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