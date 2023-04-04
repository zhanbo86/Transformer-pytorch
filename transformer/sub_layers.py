"""definition the sub layers of tranformer by SteveZhan"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

        
class EncoderLayer(nn.Module):
    """encoder layer"""
    def __init__(self, num_heads, d_k_embd, d_model, d_ff_hid, dropout=0.2) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)  ## difference: move layer norm from behind multi-head attetion to before
        self.multi_head_attention = MultiHeadAttention(num_heads, d_k_embd, d_model, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)  ## difference: move layer norm from behind FeedForward to before
        self.feed_forward = FeedForward(d_in=d_model, d_hid=d_ff_hid, dropout=dropout)
        
    def forward(self, enc_input, slf_attn_mask=None):
        """
        Args:
            enc_input is (B, T, C) = (batch_size, input_size, d_model)
            slf_attn_mask is (batch_size, 1, input_size)
        """
        enc_output = self.layer_norm_1(enc_input) # enc_input is (batch_size, input_size, d_model)
        enc_output = enc_input + self.multi_head_attention(enc_output, enc_output, enc_output, mask=slf_attn_mask) # enc_output is (batch_size, input_size, d_model)
        out = enc_output + self.feed_forward(self.layer_norm_2(enc_output)) # (batch_size, input_size, d_model)
        return out
        

class DecoderLayer(nn.Module):
    """decoder layer"""
    def __init__(self, num_heads, d_k_embd, d_model, d_ff_hid, dropout=0.2) -> None:
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.masked_multi_head_attention = MultiHeadAttention(num_heads, d_k_embd, d_model, dropout=dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dec_enc_multi_head_attention = MultiHeadAttention(num_heads, d_k_embd, d_model, dropout=dropout)
        self.layer_norm_3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_in=d_model, d_hid=d_ff_hid, dropout=dropout)
        
    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        """
        Args:
            dec_input is (B, T, C) = (batch_size, output_size, d_model).
            enc_output is (B, T, C) = (batch_size, input_size, d_model).
            slf_attn_mask is (output_size, output_size) or (batch_size, output_size, output_size).
            dec_enc_attn_mask is (batch_size, 1, input_size).
        """
        dec_output = self.layer_norm_1(dec_input)  # (batch_size, output_size, d_model)
        dec_output = dec_input + self.masked_multi_head_attention(dec_output, dec_output, dec_output, mask=slf_attn_mask) # (batch_size, output_size, d_model)
        dec_output = dec_output + self.dec_enc_multi_head_attention(self.layer_norm_2(dec_output), enc_output, enc_output, mask=dec_enc_attn_mask) # (batch_size, output_size, d_model)  
        out = dec_output + self.feed_forward(self.layer_norm_3(dec_output))  # (batch_size, output_size, d_model)
        return out
        
        
class PositionalEncoding(nn.Module):
    """positional encoding"""
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = torch.FloatTensor([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return sinusoid_table.unsqueeze(0)  # return is (1, n_position, d_hid)

    def forward(self, x):
        return self.pos_table[:, :x.size(1), :].clone().detach() # return is (1, x.size(1), d_hid)


class ScaledDotProductAttention(nn.Module):
    """"ScaledDotProductAttention"""
    def __init__(self, dropout=0.2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q is (batch_size, n_head, input/output_size, d_k)
            k is (batch_size, n_head, input/output_size, d_k)
            v is (batch_size, n_head, input/output_size, d_k)
            mask is (batch_size, 1, input_size) or (output_size, output_size) or (batch_size, output_size, output_size).
        """
        assert q.shape[3]==k.shape[3] and q.shape[3]==v.shape[3] and k.shape[2]==v.shape[2], "the dimensions of q, k, v in ScaledDotProductAttention are not matched!"
        
        wei = q @ k.transpose(-2, -1) * q.shape[-1] **-0.5 # wei is (batch, n_head, input/output_size, input/output_size)
        if mask is not None:
            wei = wei.masked_fill(mask==0, float('-inf')) # wei is (batch, n_head, input/output_size, input/output_size)
        wei = F.softmax(wei, dim=-1) 
        wei = self.dropout(wei) 
        out = wei @ v  # out is (batch, n_head, input/output_size, d_k)
        return out      
         
         
class MultiHeadAttention(nn.Module):
    """Mulit-Head Attention module"""
    def __init__(self, num_heads, d_embd, d_model, dropout=0.2) -> None:
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_embd
        self.n_heads = num_heads
        self.liner_key = nn.Linear(self.d_model, num_heads * self.d_k, bias=True)
        self.liner_query = nn.Linear(self.d_model, num_heads * self.d_k, bias=True)
        self.liner_value = nn.Linear(self.d_model, num_heads * self.d_k, bias=True)
        
        self.attn_head = ScaledDotProductAttention(dropout=dropout)
        self.linear = nn.Linear(num_heads * self.d_k, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v, mask=None):
        """
        Args:
            q is (B, T, C) = (batch_size, input/output_size, d_model)
            k is (B, T, C) = (batch_size, input/output_size, d_model)
            v is (B, T, C) = (batch_size, input/output_size, d_model)
            mask is (B, T, C) = (batch_size, 1, input_size) or (output_size, output_size) or (batch_size, output_size, output_size).
        """
        batch_size = q.size(0)
        len_q, len_k, len_v = q.size(1), k.size(1), v.size(1)
        
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.liner_query(q).view(batch_size, len_q, self.n_heads, self.d_k)   # (batch, input/output_size, n_heads, d_k)
        k = self.liner_key(k).view(batch_size, len_k, self.n_heads, self.d_k)   # (batch, input/output_size, n_heads, d_k)
        v = self.liner_value(v).view(batch_size, len_v, self.n_heads, self.d_k)   # (batch, input/output_size, n_heads, d_k)  
        
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)   # For head axis broadcasting.
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)   # For head axis broadcasting.
            else:
                raise RuntimeError("The shape of mask is not correct!")
        
        out = self.attn_head(q, k, v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        out = out.transpose(1, 2).contiguous().view(batch_size, len_q, -1)  # (batch, input/output_size, d_model)
        out = self.linear(out) # (batch, input/output_size, d_model)
        out = self.dropout(out)
        return out  # (batch, input/output_size, d_model)
        
        
class FeedForward(nn.Module):
    """Positon-wise Feed-Forward Networks"""
    def __init__(self, d_in, d_hid, dropout=0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hid),
            nn.ReLU(),
            nn.Dropout(dropout), 
            nn.Linear(d_hid, d_in),
            nn.Dropout(dropout),   
        )
        
    def forward(self, x):
        """
        Args:
            x is (B, T, C) = (batch_size, input/output_size, d_model)
        """
        out = self.net(x)
        return out # (batch_size, input/output_size, d_model)
    
    

        
        