"""build transformer model"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .sub_layers import EncoderLayer, DecoderLayer, PositionalEncoding

class Transformer(nn.Module):
    """
    Realize the tranformer network following the paper "attention is all you need" strictly except two differencies:
    1. Moving all layernorms from after sublayers to before sublayers, this accelerate training speed significantly.
    2. Addding a dropout after the last linear layer in Feed-Forward sublayer refer to the transformer implementation 
       in pytorch (see torch.nn.Transformer). The benefit is not obvious in my test.
    """
    def __init__(self, input_vocab_size, output_vocab_size, d_model, 
                 word_max_len, num_heads, d_k_embd, layers, 
                 d_ff_hid, src_pad_idx=None, trg_pad_idx=None, trg_eos_idx=None, 
                 trg_bos_idx=None, dropout=0.2, is_pos_embed=False, beam_size=None, 
                 alpha=None, device="cuda") -> None:
        super().__init__()
        self.d_model = d_model  
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_eos_idx = trg_eos_idx
        self.trg_bos_idx = trg_bos_idx
        self.word_max_len = word_max_len  # word_max_len is the max length of inputs and outputs word
        self.beam_size = beam_size
        self.alpha = alpha
        self.device = device
        self.register_buffer("tril", (torch.tril(torch.ones(word_max_len, word_max_len))).bool())  # type bool is prepare for operation "&" with self._get_pad_mask()
            
        # build encoder
        self.encoder = Encoder(input_vocab_size=input_vocab_size, d_model=d_model, 
                               word_max_len=word_max_len, num_heads=num_heads, 
                               d_k_embd=d_k_embd, layers=layers, d_ff_hid=d_ff_hid, 
                               dropout=dropout, is_pos_embd=is_pos_embed, device=device)
        
        # build decoder
        self.decoder = Decoder(output_vocab_size=output_vocab_size, d_model=d_model, 
                               word_max_len=word_max_len, num_heads=num_heads, 
                               d_k_embd=d_k_embd, layers=layers, d_ff_hid=d_ff_hid, 
                               dropout=dropout, is_pos_embed=is_pos_embed, device=device)    
            
        # ouput norm and linear
        self.layer_norm = nn.LayerNorm(d_model)   # difference: add layer norm before Linear 
        self.linear = nn.Linear(d_model, output_vocab_size) 
        
        # share weight matrix between the two embedding layers (input and output embedding) and linear transformation
        self.encoder.input_embedding.weight = self.decoder.output_embedding.weight
        self.linear.weight = self.decoder.output_embedding.weight
        
        # init parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        
    
    def forward(self, trg_seq, src_seq=None):
        """
        trg_seq is (B, T) = (batch_size, input_size),  
        src_seq is (B, T) = (batch_size, output_size)
        """
        # build mask
        src_mask = self._get_src_mask(src_seq)  
        trg_mask = self._get_trg_mask(trg_seq)
        
        #encoder and decoder
        if src_seq is not None:  # Transformer including both Encoder and Decoder
            encoder_outs = self.encoder(src_seq=src_seq, src_mask=src_mask)
            decoder_ints = self.decoder(trg_seq=trg_seq, trg_mask=trg_mask, src_mask=src_mask, enc_output=encoder_outs)
        else:  # Transformer including only Decoder
            decoder_ints = self.encoder(src_seq=trg_seq, src_mask=trg_mask)  # use encoder instead of decoder for no src_seq input
          
        out = self.layer_norm(decoder_ints) # decoder_outs is (batch, output_size, d_model)
        logits = self.linear(out) # logits is (batch, output_size, output_vocab_size)
        return logits
    
    def generate(self, x, outputs_size, max_new_tokens):
        new_x = x # x is (batch, output_size)
        for _ in range(max_new_tokens):
            x_valid = new_x[:, -outputs_size:]  # x_valid is (batch, output_num_tokens)
            logits = self(trg_seq=x_valid)  # logits is (batch, output_num_tokens, token_size)
            logits = logits[:, -1, :]  # logits is (batch, token_size), only keep the last output token
            probs = F.softmax(logits, dim=-1) # probs is (batch, token_size)
            next_idx = torch.multinomial(probs, num_samples=1)  # next_idx is (batch, 1)
            new_x = torch.cat((new_x, next_idx), dim=1)
        return new_x
    
    def _get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(-2)  # output size is (batch_size, 1, input/output_size)
    
    def _get_src_mask(self, src_seq):
        # src_seq is (batch_size, input_size)
        if self.src_pad_idx is not None:
            src_mask = self._get_pad_mask(src_seq, self.src_pad_idx)  # src_mask is (batch_size, 1, input_size)
        else:
            src_mask = None
        return src_mask
    
    def _get_trg_mask(self, trg_seq):
        # trg_seq is (batch_size, output_size)
        if self.trg_pad_idx is not None:
            trg_mask = self._get_pad_mask(trg_seq, self.trg_pad_idx) & (self.tril[:trg_seq.size(1), :trg_seq.size(1)])  # trg_mask is (batch_size, output_size, output_size)
        else:
            trg_mask = self.tril[:trg_seq.size(1), :trg_seq.size(1)]  # trg_mask is (output_size, output_size)
            
        # # for pytorch transformer mask format
        # new_trg_mask = torch.ones_like(trg_mask, dtype=torch.float)
        # new_trg_mask.masked_fill_(trg_mask, 0)
        # new_trg_mask.masked_fill_(~trg_mask, float("-inf")) 
        # new_trg_mask = new_trg_mask.repeat(num_heads, 1, 1)
           
        return trg_mask
    
    def _gen_init_state(self, src_seq, src_mask, trg_mask):
        # src_seq is (batch_size, input_size), trg_seq is (batch_size, output_size), trg_mask is (output_size, output_size)/(batch_size, output_size, output_size)
        encoder_outs = self.encoder(src_seq=src_seq, src_mask=src_mask)   # encoder_outs is (batch, input_size, d_model)           
        decoder_ints = self.decoder(trg_seq=self.init_seq, trg_mask=trg_mask, src_mask=src_mask, enc_output=encoder_outs)
        out = self.layer_norm(decoder_ints) # decoder_outs is (batch, output_size, d_model)
        dec_output_prob = F.softmax(self.linear(out), dim=-1)  # dec_output_prob is (batch, output_size, output_vocab_size)
            
        best_k_probs, best_k_idx = dec_output_prob[:, -1, :].topk(self.beam_size) # best_k_probs, best_k_idx is (batch=1, beam_size)
        scores = torch.log(best_k_probs).view(self.beam_size)  # scores is (beam_size)
        gen_seq = self.blank_seqs.clone().detach() #  (beam_size, word_max_len)
        gen_seq[:, 1] = best_k_idx[0]
        encoder_outs = encoder_outs.repeat(self.beam_size, 1, 1)  #  (beam_size, input_size, d_model)
        return scores, gen_seq, encoder_outs
    
    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(self.beam_size)

        # Include the previous scores. Acculate two loops result to find the best topk vocab
        scores = torch.log(best_k2_probs).view(self.beam_size, -1) + scores.view(self.beam_size, 1) # new scores is (beam_size, beam_size)
        
        # Get the best k candidates from k^2 candidates.
        scores, best_k_idx_in_k2 = scores.view(-1).topk(self.beam_size)   # scores is (beam_size)
 
        # Get the corresponding positions of the best k candidiates.
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // self.beam_size, best_k_idx_in_k2 % self.beam_size # row, column
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs] # best_k_idx shape is [5]

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        return gen_seq, scores
        
    
    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        assert src_seq.size(0) == 1
        self.eval()
        self.register_buffer('init_seq', torch.LongTensor([[self.trg_bos_idx]]).to(self.device))
        self.register_buffer(
            'blank_seqs', 
            torch.full((self.beam_size, self.word_max_len), self.trg_pad_idx, dtype=torch.long).to(self.device)) # (beam_size, word_max_len)
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, self.word_max_len + 1, dtype=torch.long).unsqueeze(0).to(self.device)) # (1, word_max_len)

        with torch.no_grad():
            # get init state
            src_mask = self._get_src_mask(src_seq)
            trg_mask = self._get_trg_mask(trg_seq=self.init_seq)
            
            # get init scores, gen_seq, encoder_outs
            scores, gen_seq, encoder_outs = self._gen_init_state(src_seq, src_mask, trg_mask)
            
            # loop idx
            ans_idx = 0   # default
            for step in range(2, self.word_max_len):    # decode up to max length
                trg_mask = self._get_trg_mask(trg_seq=gen_seq[:, :step])
                decoder_ints = self.decoder(trg_seq=gen_seq[:, :step], trg_mask=trg_mask, src_mask=src_mask, enc_output=encoder_outs)
                out = self.layer_norm(decoder_ints) 
                dec_output_prob = F.softmax(self.linear(out), dim=-1)  
                
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output_prob, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == self.trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, indices = self.len_map.masked_fill(~eos_locs, self.word_max_len).min(1) # seq_lens means the first trg_eos_idx in each row
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == self.beam_size:  # all rows have trg_eos_idx, it means should end the sentence
                    # TODO: Try different terminate conditions.
                    # ans_idx means the highest probality vocab considering scores and the first trg_eos_idx position
                    _, ans_idx = scores.div(seq_lens.float() ** self.alpha).max(0)  
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
        
        
class Encoder(nn.Module):
    """encoder"""
    def __init__(self, input_vocab_size, d_model, word_max_len, num_heads, d_k_embd, 
                 layers, d_ff_hid, dropout=0.2, is_pos_embd=False, device="cuda") -> None:
        super().__init__()
        
        self.is_pos_embd = is_pos_embd  # true is positional embedding, false is positional encoding
        self.d_model = d_model
        self.d_k_embd = d_k_embd
        self.device = device
        
        # input&output embedding
        self.input_embedding = nn.Embedding(input_vocab_size, d_model)  # is a table of (input_vocab_size, d_model)
        
        # input&output positional encoding
        if self.is_pos_embd: 
            self.input_pos_encoding = nn.Embedding(word_max_len, d_model) # is a table of (word_max_len, d_model)
        else:  
            self.input_pos_encoding = PositionalEncoding(d_hid=d_model, n_position=word_max_len) # is a table of (1, word_max_len, d_model)
        self.dropout_input = nn.Dropout(dropout) 
        
        # encoder layer
        self.encoder_layers = nn.ModuleList([EncoderLayer(num_heads=num_heads, d_k_embd=d_k_embd, d_model=d_model, \
            d_ff_hid=d_ff_hid, dropout=dropout) for i in range(layers)])
        
    def forward(self, src_seq=None, src_mask=None):
        """
        Args:
            src_seq is (B, T) = (batch_size, input_size) or None, 
            src_mask is (B, T) = (batch_size, 1, input_size) or None,
        """
        if src_seq is not None:  
            inputs_embd = self.input_embedding(src_seq) * self.d_model ** 0.5 # inputs_embd is (batch, input_size, d_model)
            if self.is_pos_embd:
                inputs_pos_en = self.input_pos_encoding(torch.arange(src_seq.shape[1], device=self.device)) #  inputs_pos_en is (input_size, d_model)
            else:
                inputs_pos_en = self.input_pos_encoding(src_seq) # inputs_pos_en is (1, input_size, d_model)
                
            inputs = self.dropout_input(inputs_embd + inputs_pos_en)  # inputs is (batch, input_size, d_model)
            encoder_outs = inputs
            for enc_layer in self.encoder_layers:
                encoder_outs = enc_layer(enc_input=encoder_outs, slf_attn_mask=src_mask)  # encoder_outs is (batch, input_size, d_model)
                
        else:  
            encoder_outs = None
        return encoder_outs
    
    
class Decoder(nn.Module):
    """decoder"""
    def __init__(self, output_vocab_size, d_model, word_max_len, num_heads, d_k_embd, 
                 layers, d_ff_hid, dropout=0.2, is_pos_embed=False, device="cuda") -> None:
        super().__init__()
        self.d_model = d_model  
        self.layers = layers
        self.is_pos_embd = is_pos_embed   # true is positional embedding, false is positional encoding
        self.device = device
        
        # output embedding
        self.output_embedding = nn.Embedding(output_vocab_size, d_model)  # is a table of (output_vocab_size, d_model)
        
        # output positional encoding
        if is_pos_embed: 
            self.output_pos_encoding = nn.Embedding(word_max_len, d_model) # is a table of (word_max_len, d_model)
        else:  
            self.output_pos_encoding = PositionalEncoding(d_hid=d_model, n_position=word_max_len) # is a table of (1, word_max_len, d_model)
        self.dropout_output = nn.Dropout(dropout)  
        
        # decoder
        self.decoder_layers = nn.ModuleList([DecoderLayer(num_heads=num_heads, d_k_embd=d_k_embd, d_model=d_model, \
            d_ff_hid=d_ff_hid, dropout=dropout) for i in range(layers)])
        
    def forward(self, trg_seq, trg_mask, src_mask=None, enc_output=None):
        """
        Args:
            trg_seq is (batch_size, output_size)  
            trg_mask is (output_size, output_size) or (batch_size, output_size, output_size)
            src_mask is (batch_size, 1, input_size) or None
            enc_output is (batch_size, input_size, d_model), 
        """
        outputs_embd = self.output_embedding(trg_seq) * self.d_model ** 0.5 # outputs_embd is (batch, output_size, d_model)
        if self.is_pos_embd:
            outputs_pos_en = self.output_pos_encoding(torch.arange(trg_seq.shape[1], device=self.device)) #  outputs_pos_en is (output_size, d_model)
        else:
            outputs_pos_en = self.output_pos_encoding(trg_seq) # outputs_pos_en is (1, output_size, d_model)
        
        outputs = self.dropout_output(outputs_embd + outputs_pos_en)  # outputs is (batch, output_size, d_model)
        decoder_ints = outputs
        for dec_layer in self.decoder_layers:
            decoder_ints = dec_layer(dec_input=decoder_ints, enc_output=enc_output, 
                                     slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask) 
               
        return decoder_ints # decoder_ints is (batch, output_size, d_model)
        

    