"""train entry for seq2seq by SteveZhan"""

import os
import sys
import argparse
import torch
import torch.optim as optim
import time
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from transformer.transformer import Transformer
from transformer.optim import ScheduledOptim
from dataloader.shakespear_dataset import ShakespeareDataset
from dataloader.wmt16_dataset import prepare_dataloaders


class TrainerDecoder():
    """trainer for a uncompleted transformer model only including decoder"""
    def __init__(self):
        # parameters for shakespear_dataset
        self.batch_size = 64
        self.output_size = 256    ## output block_size
        self.word_max_len = 300   ## word_max_len must be larger than output_size/input_size
        self.d_model = 384
        self.d_ff_hid = self.d_model * 4  #384 * 4
        self.num_heads = 6
        self.d_k_embd = self.d_model//self.num_heads
        self.layers = 6
        self.dropout = 0.2
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 3e-4  # 3e-4
        self.eval_iters = 200
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # create ShakespeareDataset dataset
        file_path = "./dataloader/input.txt"
        self.dataset = ShakespeareDataset(file_path)
        self.vocab_size = self.dataset.vocab_size

        # create model
        self.model = Transformer(
            input_vocab_size = self.vocab_size, 
            output_vocab_size = self.vocab_size,
            d_model = self.d_model, 
            word_max_len = self.word_max_len,
            num_heads = self.num_heads, 
            d_k_embd = self.d_k_embd, 
            layers = self.layers, 
            d_ff_hid = self.d_ff_hid,
            dropout = self.dropout)
        self.model.to(self.device)
        
        # create optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
    
    def train(self):
        # start iteration
        self.model.train()
        for iter in range(self.max_iters):
            if iter % self.eval_interval == 0:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
            xb, yb = self.dataset.get_batch('train', self.batch_size, self.output_size)
            xb, yb = xb.to(self.device), yb.to(self.device)
            pred = self.model(trg_seq=xb)
            
            # calculate loss
            pred = pred.view(-1, pred.size(2))  # pred is (batch * output_num_tokens, token_size)
            targets = yb.view(yb.size(0)*yb.size(1))  # targets is (batch * output_num_tokens)
            loss = F.cross_entropy(pred, targets) 
            
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        losses = self.estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    #define estimation module in training process
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.dataset.get_batch(split, self.batch_size, self.output_size)
                X, Y = X.to(self.device), Y.to(self.device), 
                pred = self.model(trg_seq=X)
                
                # calculate loss
                pred = pred.view(-1, pred.size(2))  # pred is (batch * output_num_tokens, token_size)
                targets = Y.view(Y.size(0)*Y.size(1))  # targets is (batch * output_num_tokens)
                loss = F.cross_entropy(pred, targets)                 
                losses[k] = loss.item()
                
            out[split] = losses.mean()
        self.model.train()
        return out


class TrainerTransformer():
    """trainder for a completed transformer, including encoder and decoder"""
    def __init__(self):
        # parameters
        self.batch_size = 128 #128
        self.word_max_len = 300   ## word_max_len must be larger than output_size/input_size
        self.d_model = 512
        self.d_ff_hid = 512 * 4
        self.num_heads = 8
        self.d_k_embd = self.d_model//self.num_heads
        self.layers = 6
        self.dropout = 0.1
        self.epoch = 80
        self.lr_mul = 0.5 #2.0
        self.n_warmup_steps = 4000 #128000
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.output_dir = "./output"
        self.tb_writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))
        self.label_smoothing = True
        self.max_len = 100
        self.min_freq = 3
        self.is_pos_enbd = False
        
        # for translate
        self.alpha = 0.7
        self.beam_size = 5
        self.max_seq_len = 100
        
        # create WMT16 dataset
        self.training_data, self.validation_data, self.test_dataset, self.train_datasets_len, self.val_datasets_len, self.src_pad_idx, self.trg_pad_idx, \
             self.trg_bos_idx, self.trg_eos_idx, self.src_vocab_size, self.trg_vocab_size, self.unk_idx, \
             self.src_stoi, self.trg_itos, self.BOS_WORD, self.EOS_WORD  \
             = prepare_dataloaders(self.max_len, self.min_freq, self.batch_size, self.device)
        self.train_datasets_iter = iter(self.training_data)
        self.val_datasets_iter = iter(self.validation_data)
        
        # create model
        self.model = Transformer(
            input_vocab_size = self.src_vocab_size,
            output_vocab_size = self.trg_vocab_size,  
            d_model = self.d_model, 
            word_max_len = self.word_max_len,
            num_heads = self.num_heads, 
            d_k_embd = self.d_k_embd, 
            layers = self.layers, 
            d_ff_hid = self.d_ff_hid,
            src_pad_idx = self.src_pad_idx, 
            trg_pad_idx = self.trg_pad_idx, 
            trg_eos_idx = self.trg_eos_idx, 
            trg_bos_idx = self.trg_bos_idx,
            dropout = self.dropout,
            is_pos_embed = self.is_pos_enbd,
            beam_size = self.beam_size,
            alpha = self.alpha,
            device = self.device)
        self.model.to(self.device)
        
        # create optimizer
        self.optimizer = ScheduledOptim(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            self.lr_mul, self.d_model, self.n_warmup_steps)
        
    def train(self):
        valid_losses = []
        for epoch_i in range(self.epoch):
            
            # train epoch
            print('[ Epoch', epoch_i, ']')
            start = time.time()
            train_loss, train_accu = self.train_epoch()
            train_ppl = math.exp(min(train_loss, 100))
            lr = self.optimizer._optimizer.param_groups[0]['lr']
            self.print_performances('Training', train_ppl, train_accu, start, lr)
            
            # eval epoch
            start = time.time()
            valid_loss, valid_accu = self.eval_epoch()
            valid_ppl = math.exp(min(valid_loss, 100))
            self.print_performances('Validation', valid_ppl, valid_accu, start, lr)

            valid_losses += [valid_loss]
            checkpoint = {'epoch': epoch_i, 'model': self.model.state_dict()}
            if valid_loss <= min(valid_losses):
                torch.save(checkpoint, os.path.join(self.output_dir, 'model.chkpt'))
                print('    - [Info] The checkpoint file has been updated.')

            # write to tensorboard
            self.tb_writer.add_scalars('ppl', {'train': train_ppl, 'val': valid_ppl}, epoch_i)
            self.tb_writer.add_scalars('accuracy', {'train': train_accu*100, 'val': valid_accu*100}, epoch_i)
            self.tb_writer.add_scalar('learning_rate', lr, epoch_i) 
        
    def train_epoch(self):
        ''' Epoch operation in training phase'''

        self.model.train()
        total_loss, n_word_total, n_word_correct = 0, 0, 0 
        print("train epoch is processing: ")
        
        for i in range(self.train_datasets_len//self.batch_size):
            # prepare data
            batch = next(self.train_datasets_iter)
            src_seq = self._patch_src(batch['src'], self.src_pad_idx).to(self.device) # src_seq is (batch_size, output_size-1)
            trg_seq, gold = map(lambda x: x.to(self.device), self._patch_trg(batch['trg'], self.trg_pad_idx)) 
            # trg_seq is (batch_size, output_size-1), gold is (batch_size * (output_size-1))

            # forward
            self.optimizer.zero_grad()
            pred = self.model(src_seq=src_seq, trg_seq=trg_seq)
            pred = pred.view(-1, pred.size(2))

            # backward and update parameters
            loss, n_correct, n_word = self.cal_performance(
                pred, gold, self.trg_pad_idx, smoothing=self.label_smoothing) 
            loss.backward()
            self.optimizer.step_and_update_lr()

            # note keeping
            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += loss.item()

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy

    def eval_epoch(self):
        ''' Epoch operation in evaluation phase '''
        self.model.eval()
        total_loss, n_word_total, n_word_correct = 0, 0, 0
        print("val epoch is processing: ")

        with torch.no_grad():
            for i in range(self.val_datasets_len//self.batch_size):
                # prepare data
                batch = next(self.val_datasets_iter)
                src_seq = self._patch_src(batch['src'], self.src_pad_idx).to(self.device)
                trg_seq, gold = map(lambda x: x.to(self.device), self._patch_trg(batch['trg'], self.trg_pad_idx))

                # forward
                pred = self.model(src_seq=src_seq, trg_seq=trg_seq)
                pred = pred.view(-1, pred.size(2))
                loss, n_correct, n_word = self.cal_performance(
                    pred, gold, self.trg_pad_idx, smoothing=False)

                # note keeping
                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss.item()

        loss_per_word = total_loss/n_word_total
        accuracy = n_word_correct/n_word_total
        return loss_per_word, accuracy
    
    def _patch_src(self, src, pad_idx):
        src = src.transpose(0, 1)
        return src
    
    def _patch_trg(self, trg, pad_idx):
        trg = trg.transpose(0, 1)
        trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
        return trg, gold
    
    def print_performances(self, header, ppl, accu, start_time, lr):
        print('  - {header:12} ppl: {ppl: 8.5f}, accuracy: {accu:3.3f} %, lr: {lr:8.5f}, '\
            'elapse: {elapse:3.3f} min'.format(
                header=f"({header})", ppl=ppl,
                accu=100*accu, elapse=(time.time()-start_time)/60, lr=lr))  
    
    def cal_loss(self, pred, gold, trg_pad_idx, smoothing=False):
        ''' Calculate cross entropy loss, apply label smoothing if needed. '''
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            non_pad_mask = gold.ne(trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')
        return loss
    
    def cal_performance(self, pred, gold, trg_pad_idx, smoothing=False):
        ''' Apply label smoothing if needed '''

        loss = self.cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

        pred = pred.max(1)[1]
        # print(pred.shape, " pred after: ", pred)
        
        gold = gold.contiguous().view(-1)
        # print(gold.shape, " gold: ", gold)
        
        non_pad_mask = gold.ne(trg_pad_idx)
        n_correct = pred.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()
        return loss, n_correct, n_word
    
    def translate(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')
        
        for src, trg in tqdm(self.test_dataset, mininterval=2, desc='  - (Test)', leave=False):
            print("")
            print("German input: ", "[", len(src), "]", ' '.join(src))
            print("English gt: ", "[", len(trg), "]", ' '.join(trg))
            src_seq = [self.src_stoi.get(word, self.unk_idx) for word in src]
            pred_seq = self.model.translate_sentence(torch.LongTensor([src_seq]).to(self.device))
            pred_line = ' '.join(self.trg_itos[idx] for idx in pred_seq)
            pred_line = pred_line.replace(self.BOS_WORD, '').replace(self.EOS_WORD, '')
            print("English pred: ", "[", len(pred_seq), "]", pred_seq)
            print("English pred: ", "[", len(pred_seq), "]", pred_line)
        print('[Info] Finished.')

 
def main():
    """
    usage:
    python train.py --net transformer or decoder
    "decoder" is a uncompleted transformer model only including decoder
    "transformer" is a completed transformer, including encoder and decoder
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--net', type=str, choices=['transformer', 'decoder'], default='transformer')
    opt = parser.parse_args()
    print("opt", opt)
    
    torch.manual_seed(1337)
    
    if opt.net == "decoder":
        trainer =TrainerDecoder()
        trainer.train()
        context = torch.zeros((1,1), dtype=torch.long, device=trainer.device)   #input idx is (1, 1)
        print(trainer.dataset.decode(trainer.model.generate(context, outputs_size=256, max_new_tokens=500)[0].tolist()))
    elif opt.net == "transformer":
        trainer = TrainerTransformer()
        trainer.train()
        trainer.translate(model_path="./output/model.chkpt")
    else:
        print("unsupported net!")
    
if __name__ == "__main__":
    main()
    
