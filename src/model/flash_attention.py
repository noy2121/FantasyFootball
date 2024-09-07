import torch
import torch.nn as nn
from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input


class FlashAttentionWrapper(nn.Module):
    def __init__(self, attention_module):
        super().__init__()
        self.attention_module = attention_module

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # Get query, key, value projections
        query, key, value = self.attention_module.get_qkv(hidden_states)

        # Prepare inputs for flash attention
        qkv = torch.stack([query, key, value], dim=2)
        batch_size, seq_len, _, num_heads, head_dim = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, num_heads * head_dim)

        # Handle attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(1).squeeze(1)
            qkv, indices, cu_seqlens, max_seqlen = unpad_input(qkv, attention_mask)
            output = flash_attn_qkvpacked_func(qkv, cu_seqlens, max_seqlen, 0.0, softmax_scale=None, causal=False)
            output = pad_input(output, indices, batch_size, seq_len)
        else:
            output = flash_attn_qkvpacked_func(qkv, None, None, 0.0, softmax_scale=None, causal=False)

        output = output.reshape(batch_size, seq_len, num_heads, head_dim)
        output = self.attention_module.dense(output)

        return (output, None) if output_attentions else (output,)


def apply_flash_attention(model):
    for name, module in model.named_modules():
        if "attention" in name.lower() and hasattr(module, 'get_qkv'):
            setattr(model, name, FlashAttentionWrapper(module))
    return model