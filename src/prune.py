import torch
import torch_pruning as tp
import torch.nn.utils.prune as prune


def apply_prune(model):
    p = prune.Identity()
    p.apply(model.embeddings.word_embeddings, 'weight')
    p.apply(model.embeddings.position_embeddings, 'weight')
    p.apply(model.embeddings.token_type_embeddings, 'weight')
    p.apply(model.embeddings.LayerNorm, 'weight')
    p.apply(model.embeddings.LayerNorm, 'bias')
    
    for layer in model.encoder.layer:
        p.apply(layer.attention.self.query, 'weight')
        p.apply(layer.attention.self.query, 'bias')
        p.apply(layer.attention.self.key, 'weight')
        p.apply(layer.attention.self.key, 'bias')
        p.apply(layer.attention.self.value, 'weight')
        p.apply(layer.attention.self.value, 'bias')
        
        p.apply(layer.attention.output.dense, 'weight')
        p.apply(layer.attention.output.dense, 'bias')
        p.apply(layer.attention.output.LayerNorm, 'weight')
        p.apply(layer.attention.output.LayerNorm, 'bias')
        
        p.apply(layer.intermediate.dense, 'weight')
        p.apply(layer.intermediate.dense, 'bias')
        
        p.apply(layer.output.dense, 'weight')
        p.apply(layer.output.dense, 'bias')
        p.apply(layer.output.LayerNorm, 'weight')
        p.apply(layer.output.LayerNorm, 'bias')


def remove_prune(model):
    prune.remove(model.embeddings.word_embeddings, 'weight')
    prune.remove(model.embeddings.position_embeddings, 'weight')
    prune.remove(model.embeddings.token_type_embeddings, 'weight')
    prune.remove(model.embeddings.LayerNorm, 'weight')
    prune.remove(model.embeddings.LayerNorm, 'bias')
    
    for layer in model.encoder.layer:
        prune.remove(layer.attention.self.query, 'weight')
        prune.remove(layer.attention.self.query, 'bias')
        prune.remove(layer.attention.self.key, 'weight')
        prune.remove(layer.attention.self.key, 'bias')
        prune.remove(layer.attention.self.value, 'weight')
        prune.remove(layer.attention.self.value, 'bias')
        
        prune.remove(layer.attention.output.dense, 'weight')
        prune.remove(layer.attention.output.dense, 'bias')
        prune.remove(layer.attention.output.LayerNorm, 'weight')
        prune.remove(layer.attention.output.LayerNorm, 'bias')
        
        prune.remove(layer.intermediate.dense, 'weight')
        prune.remove(layer.intermediate.dense, 'bias')
        
        prune.remove(layer.output.dense, 'weight')
        prune.remove(layer.output.dense, 'bias')
        prune.remove(layer.output.LayerNorm, 'weight')
        prune.remove(layer.output.LayerNorm, 'bias')
        
        
        
def apply_mask(model, hidden_idxs, attn_idxs, intermediate_idxs, val=0.):
    model.embeddings.word_embeddings.weight_mask[:, hidden_idxs] = val
    model.embeddings.position_embeddings.weight_mask[:, hidden_idxs] = val
    model.embeddings.token_type_embeddings.weight_mask[:, hidden_idxs] = val
    model.embeddings.LayerNorm.weight_mask[hidden_idxs] = val
    model.embeddings.LayerNorm.bias_mask[hidden_idxs] = val
    
    for layer in model.encoder.layer:
        layer.attention.self.query.weight_mask[:, hidden_idxs] = val
        layer.attention.self.query.weight_mask[attn_idxs] = val
        layer.attention.self.query.bias_mask[attn_idxs] = val

        layer.attention.self.key.weight_mask[:, hidden_idxs] = val
        layer.attention.self.key.weight_mask[attn_idxs] = val
        layer.attention.self.key.bias_mask[attn_idxs] = val

        layer.attention.self.value.weight_mask[:, hidden_idxs] = val
        layer.attention.self.value.weight_mask[attn_idxs] = val
        layer.attention.self.value.bias_mask[attn_idxs] = val 
        
        layer.attention.output.dense.weight_mask[:, attn_idxs] = val
        layer.attention.output.dense.weight_mask[hidden_idxs] = val
        layer.attention.output.dense.bias_mask[hidden_idxs] = val

        layer.attention.output.LayerNorm.weight_mask[hidden_idxs] = val
        layer.attention.output.LayerNorm.bias_mask[hidden_idxs] = val
        
        layer.intermediate.dense.weight_mask[:, hidden_idxs] = val
        layer.intermediate.dense.weight_mask[intermediate_idxs] = val
        layer.intermediate.dense.bias_mask[intermediate_idxs] = val
        
        layer.output.dense.weight_mask[:, intermediate_idxs] = val
        layer.output.dense.weight_mask[hidden_idxs] = val
        layer.output.dense.bias_mask[hidden_idxs] = val

        layer.output.LayerNorm.weight_mask[hidden_idxs] = val
        layer.output.LayerNorm.bias_mask[hidden_idxs] = val
        
        
def reset_mask(model):
    model.embeddings.word_embeddings.weight_mask.fill_(1.)
    model.embeddings.position_embeddings.weight_mask.fill_(1.)
    model.embeddings.token_type_embeddings.weight_mask.fill_(1.)
    model.embeddings.LayerNorm.weight_mask.fill_(1.)
    model.embeddings.LayerNorm.bias_mask.fill_(1.)
    
    for layer in model.encoder.layer:
        layer.attention.self.query.weight_mask.fill_(1.)
        layer.attention.self.query.bias_mask.fill_(1.)
        layer.attention.self.key.weight_mask.fill_(1.)
        layer.attention.self.key.bias_mask.fill_(1.)
        layer.attention.self.value.weight_mask.fill_(1.)
        layer.attention.self.value.bias_mask.fill_(1.) 
        
        layer.attention.output.dense.weight_mask.fill_(1.)
        layer.attention.output.dense.bias_mask.fill_(1.)
        layer.attention.output.LayerNorm.weight_mask.fill_(1.)
        layer.attention.output.LayerNorm.bias_mask.fill_(1.)
        
        layer.intermediate.dense.weight_mask.fill_(1.)
        layer.intermediate.dense.bias_mask.fill_(1.)
        
        layer.output.dense.weight_mask.fill_(1.)
        layer.output.dense.bias_mask.fill_(1.)
        layer.output.LayerNorm.weight_mask.fill_(1.)
        layer.output.LayerNorm.bias_mask.fill_(1.)


def remove_weight(model, hidden_idxs, attn_idxs, intermediate_idxs):
    graph = tp.DependencyGraph()
    graph.build_dependency(model, example_inputs=torch.randint(0, 10000, (1, 128)), pruning_dim=-1)

    # hidden idxs
    plan = graph.get_pruning_plan(model.embeddings.word_embeddings, tp.prune_embedding, idxs=hidden_idxs)
    plan.exec()
    if model.pooler is not None:
        plan = graph.get_pruning_plan(model.pooler.dense, tp.prune_linear, idxs=hidden_idxs)
        plan.exec()

    # attn, intermeidate idxs
    for layer in model.encoder.layer:
        plan = graph.get_pruning_plan(layer.attention.self.query, tp.prune_linear, idxs=attn_idxs)
        plan.exec()
        layer.attention.self.all_head_size -= len(attn_idxs)
        layer.attention.self.attention_head_size = layer.attention.self.all_head_size  // layer.attention.self.num_attention_heads
        
        plan = graph.get_pruning_plan(layer.intermediate.dense, tp.prune_linear, idxs=intermediate_idxs)
        plan.exec()

    # config
    model.config.hidden_size -= len(hidden_idxs)
    model.config.intermediate_size -= len(intermediate_idxs)
    return model