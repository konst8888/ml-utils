import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import random

class ELRLoss(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3, lam=0.01, weight=None, device='cpu', fix=False):
        super(ELRLoss, self).__init__()
        self.num_classes = num_classes
        self.target = {
            'train': torch.zeros(num_examp['train'], self.num_classes).to(device),
            'valid': torch.zeros(num_examp['valid'], self.num_classes).to(device),
        }
        self.phase = 'train'
        self.beta = beta
        self.lam = lam
        self.weight_kw = {} if weight is None else {'weight': weight}
        self.device = device
        self.fix = fix
        
    def set_phase(self, phase):
        self.phase = phase

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[self.phase][index] = self.beta * self.target[self.phase][index] + (1-self.beta) * ((y_pred_)/(y_pred_).sum(dim=1,keepdim=True))        
        self.target[self.phase][index] = y_pred_
        
        if False and torch.any(label[:10] == 3):
            print(label[:10])
            print(y_pred[:10])
            print(self.target[self.phase][index][:10])
            import sys
            sys.exit()
        ce_loss = F.cross_entropy(output, label, **self.weight_kw)
        elr_reg = ((1-(self.target[self.phase][index] * y_pred).sum(dim=1)).log()).mean()
        elr_reg *= self.lam
        final_loss = ce_loss + elr_reg
        return  final_loss, float(elr_reg.item())

class BiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, total_words, num_class, fixed_embeds=None, pretrained = False, pretrained_embed = None):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        #self.wordembed = nn.Embedding.from_pretrained(pretrained_embed, freeze = False)
        if fixed_embeds is None:
            self.wordembed = nn.Embedding(total_words, embedding_size)
        else:
            self.wordembed = FixedEmbedding(fixed_embeds, total_words, embedding_size)
        # self.fcembed = nn.Linear(embedding_size, embedding_size)
        # self.for_charembed = forwardLSTM()
        # self.back_charembed = bachwardLSTM()
        self.dropout = nn.Dropout(p = 0.5) # 0.5
        self.bilstm = nn.LSTM(
            embedding_size,
            hidden_size, 
            num_layers=1, 
            bidirectional = True, 
            batch_first = True,
            #dropout=0.5
        )
        self.linear = nn.Linear(2*hidden_size, num_class) # 2 because forward and backward concatenate

    def forward(self, x): #add xchar
        #xlengths = torch.LongTensor([len(y) for y in x])
        #x = pack_padded_sequence(x, xlengths.cpu(), batch_first=True, enforce_sorted=False)
        #x, _ = pad_packed_sequence(x, batch_first=True)

        word_embedding = self.wordembed(x) # x is of size(batchsize, seq_len), out is of size (batchsize, seq_len, embedding_size = 100)
        # word_embedding = self.fcembed(word_embedding)
        word_embedding = self.dropout(word_embedding) # dropout

        out, (h,c) = self.bilstm(word_embedding) #'out' has dimension(batchsize, seq_len, 2*hidden_size)
        #h = h.view(h.shape[1], h.shape[0]*h.shape[2])
        h = self.dropout(torch.cat((h[-2,:,:], h[-1,:,:]), dim = 1))
        out = self.linear(h)
        #out = self.linear(out[:, -1:, ...]) # now 'out' has dimension(batchsize, seq_len, num_class)
        #out = out.view(-1, out.shape[2]) # shape (128*seqlen, 18)
        #out = F.log_softmax(out, dim=1) # take the softmax across the dimension num_class, 'out' has dimension(batchsize, seq_len, num_class)
        #print(out.shape)
        #out = out.view(out.shape[0], out.shape[2])
        return out

    
class AttnBiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, total_words, num_class, fixed_embeds=None, pretrained = False, pretrained_embed = None):
        
        super(AttnBiLSTM, self).__init__()
        
        self.input_size = embedding_size
        self.hidden_size = hidden_size
        
        self.wordembed = nn.Embedding(total_words, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers = 1, bidirectional = True)
        self.attention_weights_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p = 0.5)
        self.linear = nn.Linear(hidden_size, num_class)
        self.act_func = nn.Softmax(dim=1)
    
    def forward(self, x, lengths):
        # [seq_len, batch, input_size]
        #x [batch_size, sentence_length, embedding_size]
        x = self.wordembed(x) # x is of size(batchsize, seq_len), out is of size (batchsize, seq_len, embedding_size = 100)
        # word_embedding = self.fcembed(word_embedding)
        x = self.dropout(x) # dropout
        
        x = x.permute(1, 0, 2)         #[sentence_length, batch_size, embedding_size]
        
        batch_size = x.size(1)
        
        seq_len, batch_size, _ = x.size()
        x = pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)
        
        #h_0 = torch.randn(3 * 2, batch_size, self.hidden_size) #.to(self.device)
        #c_0 = torch.randn(3 * 2, batch_size, self.hidden_size) #.to(self.device)
        
        #out[seq_len, batch, num_directions * hidden_size]
        #h_n, c_n [num_layers * num_directions, batch, hidden_size]
        out, (h_n, c_n) = self.lstm(
            x, 
            #(h_0, c_0)
        )
        
        out, _ = pad_packed_sequence(out, batch_first=False)
        #x = x.contiguous()
        #x = x.view(-1, x.shape[2])
        
        (forward_out, backward_out) = torch.chunk(out, 2, dim = 2)
        out = forward_out + backward_out  #[seq_len, batch, hidden_size]
        out = out.permute(1, 0, 2)  #[batch, seq_len, hidden_size]
        
        h_n = h_n.permute(1, 0, 2)  #[batch, num_layers * num_directions,  hidden_size]
        h_n = torch.sum(h_n, dim=1) #[batch, 1,  hidden_size]
        h_n = h_n.squeeze(dim=1)  #[batch, hidden_size]
        
        attention_w = self.attention_weights_layer(h_n)  #[batch, hidden_size]
        attention_w = attention_w.unsqueeze(dim=1) #[batch, 1, hidden_size]
        
        attention_context = torch.bmm(attention_w, out.transpose(1, 2))  #[batch, 1, seq_len]
        softmax_w = F.softmax(attention_context, dim=-1)  #[batch, 1, seq_len]
        
        #out = self.dropout(out)
        x = torch.bmm(softmax_w, out)  #[batch, 1, hidden_size]
        x = x.squeeze(dim=1)  #[batch, hidden_size]
        #x = self.dropout(x)
        x = self.linear(x)
        #x = self.act_func(x)
        
        return x
