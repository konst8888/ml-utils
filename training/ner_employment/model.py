import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import random

#from qrnn.qrnn import QRNN

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
        """
        true_classes = torch.LongTensor([0, 1, 2, 4, 5])
        mask = torch.zeros(self.target[self.phase][index].shape, device=self.device, dtype=torch.bool)
        mask[:, true_classes] = True
        #self.target[self.phase][index] = self.target[self.phase][index].masked_scatter(mask, torch.zeros(*mask.shape).to(self.device))
        self.target[self.phase][index] = self.target[self.phase][index].masked_scatter(mask, y_pred_)
        
        self.target[self.phase][index] = torch.where(
            label.unsqueeze(1) == 0, 
            self.target[self.phase][index], 
            self.target[self.phase][index] * 0
        )
        """
        if not self.fix:
            true_classes = torch.LongTensor([0, 1, 2, 4, 5]).to(self.device)
            all_classes = torch.arange(6).to(self.device)
            self.target[self.phase][index] = torch.where(
                (label.unsqueeze(1) != 0) | (all_classes[..., None] == true_classes).any(-1), 
                y_pred_,
                self.target[self.phase][index],
            )
        else:
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

class ELRLoss1(nn.Module):
    def __init__(self, num_examp, num_classes=10, beta=0.3, lam=0.01, weight=None, device='cpu'):
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
        
    def set_phase(self, phase):
        self.phase = phase

    def forward(self, index, output, label):
        y_pred = F.softmax(output,dim=1)
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[self.phase][index] = self.beta * self.target[self.phase][index] + (1-self.beta) * label
        ce_loss = F.cross_entropy(output, self.target, **self.weight_kw)
        return  ce_loss


class FixedEmbedding(nn.Module):
    
    def __init__(self, fixed_embeds, num_embeddings, embedding_dim):
        super(FixedEmbedding, self).__init__()
        
        weights_freeze = np.vstack([v for k, v in fixed_embeds.items()])
        weights_freeze = torch.from_numpy(weights_freeze)
        weights_freeze = Parameter(weights_freeze, requires_grad=True)
        weights_train = Parameter(torch.randn(num_embeddings - len(fixed_embeds), embedding_dim), requires_grad=True)
        #weights = torch.cat((weights_freeze, weights_train), 0)
        weights = torch.randn(num_embeddings, embedding_dim)
        #weights[: len(fixed_embeds), :] = weights_freeze
        #weights[len(fixed_embeds): , :] = weights_train
        #self.weights = weights.float()
        
        self.weights_freeze = weights_freeze
        self.weights_train = weights_train      
            
        from copy import deepcopy
        self.old_weights = deepcopy(weights_freeze.float().detach().numpy()), deepcopy(weights_train.float().detach().numpy())

    def forward(self, idx):
        #print(idx.device)
        #print(self.weights.device)
        if idx.device != self.weights_freeze.device:
            #self.weights = self.weights.to(idx.device)
            self.weights_freeze = self.weights_freeze.to(idx.device)
            self.weights_train = self.weights_train.to(idx.device)        
        
        weights = torch.cat((self.weights_freeze, self.weights_train), 0).float()
        
        if random.random() < 0.05:
            pass
            #print(weights[0, :20])
            #print(weights[-1, :20])
            #print(np.all(self.weights_freeze.cpu().detach().numpy() == self.old_weights[0]))
            #print(np.all(self.weights_train.cpu().detach().numpy() == self.old_weights[1]))
        
        return F.embedding(idx, weights)

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
        self.bilstm = nn.LSTM(embedding_size,hidden_size, num_layers=1, bidirectional = True, batch_first = True) # bidirectional = True
        self.linear = nn.Linear(2*hidden_size, num_class) # 2 because forward and backward concatenate

    def forward(self, x): #add xchar
        #xlengths = torch.LongTensor([len(y) for y in x])
        #x = pack_padded_sequence(x, xlengths.cpu(), batch_first=True, enforce_sorted=False)
        #x, _ = pad_packed_sequence(x, batch_first=True)

        word_embedding = self.wordembed(x) # x is of size(batchsize, seq_len), out is of size (batchsize, seq_len, embedding_size = 100)
        # word_embedding = self.fcembed(word_embedding)
        word_embedding = self.dropout(word_embedding) # dropout

        out, (h,c) = self.bilstm(word_embedding) #'out' has dimension(batchsize, seq_len, 2*hidden_size)
        out = self.linear(out) # now 'out' has dimension(batchsize, seq_len, num_class)
        #out = out.view(-1, out.shape[2]) # shape (128*seqlen, 18)
        #out = F.log_softmax(out, dim=1) # take the softmax across the dimension num_class, 'out' has dimension(batchsize, seq_len, num_class)
        out = out.view(out.shape[0], out.shape[2], out.shape[1])
        return out


class BiQRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_class, device='cpu', pretrained = False, pretrained_embed = None):
        super(BiQRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = 0.5)
        """
        self.qrnn = QRNN(
            embedding_size, 
            hidden_size, 
            batch_first = False,
            num_layers=2,
            dropout=0,
            use_cuda= ('cuda' in device),
            
            zoneout=0,
            window=1,
            save_prev_x=False,
        )
        """
        # можно добавить еще один linear перед тем как подать в lstm чтобы похожие слова имели похожие вектора
        self.qrnn = nn.LSTM(2*embedding_size, hidden_size, num_layers=2, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(2*hidden_size, num_class)
        self.transform = nn.Linear(embedding_size, 2*embedding_size)

    def forward(self, x): #add xchar
        #xlengths = torch.LongTensor([len(y) for y in x])
        #x = pack_padded_sequence(x, xlengths.cpu(), batch_first=True, enforce_sorted=False)
        #x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.transform(x) # needs dropout after transforming 
        #x = self.dropout(x) # dropout
        out, h = self.qrnn(x) #'out' has dimension(batchsize, seq_len, 2*hidden_size)
        #out = out.permute(1, 0, 2)
        out = self.linear(out) # now 'out' has dimension(batchsize, seq_len, num_class)
        # maybe add relu here
        #out = out.view(-1, out.shape[2]) # shape (128*seqlen, 18)
        #out = F.log_softmax(out, dim=1) # take the softmax across the dimension num_class, 'out' has dimension(batchsize, seq_len, num_class)
        out = out.view(out.shape[0], out.shape[2], out.shape[1])
        return out

class BiQRNN1(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_class, device='cpu', pretrained = False, pretrained_embed = None):
        super(BiQRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p = 0.5)
        """
        self.qrnn = QRNN(
            embedding_size, 
            hidden_size, 
            batch_first = False,
            num_layers=2,
            dropout=0,
            use_cuda= ('cuda' in device),
            
            zoneout=0,
            window=1,
            save_prev_x=False,
        )
        """
        # можно добавить еще один linear перед тем как подать в lstm чтобы похожие слова имели похожие вектора
        self.qrnn = nn.LSTM(embedding_size, hidden_size, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(hidden_size, num_class)

    def forward(self, x): #add xchar
        #xlengths = torch.LongTensor([len(y) for y in x])
        #x = pack_padded_sequence(x, xlengths.cpu(), batch_first=True, enforce_sorted=False)
        #x, _ = pad_packed_sequence(x, batch_first=True)

        #x = self.dropout(x) # dropout
        out, h = self.qrnn(x) #'out' has dimension(batchsize, seq_len, 2*hidden_size)
        #out = out.permute(1, 0, 2)
        out = self.linear(out) # now 'out' has dimension(batchsize, seq_len, num_class)
        # maybe add relu here
        #out = out.view(-1, out.shape[2]) # shape (128*seqlen, 18)
        #out = F.log_softmax(out, dim=1) # take the softmax across the dimension num_class, 'out' has dimension(batchsize, seq_len, num_class)
        out = out.view(out.shape[0], out.shape[2], out.shape[1])
        return out

class NoiseBiLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_class, noisy_classes=[0], infer=False, device='cpu', pretrained = False, pretrained_embed = None):
        super(NoiseBiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.noisy_classes = noisy_classes
        self.infer = infer
        self.dropout = nn.Dropout(p = 0.5)
        # можно добавить еще один linear перед тем как подать в lstm чтобы похожие слова имели похожие вектора
        self.qrnn = nn.LSTM(2*embedding_size, hidden_size, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(2*hidden_size, num_class)
        self.transform = nn.Linear(embedding_size, 2*embedding_size)

    def forward(self, x): #add xchar
        #xlengths = torch.LongTensor([len(y) for y in x])
        #x = pack_padded_sequence(x, xlengths.cpu(), batch_first=True, enforce_sorted=False)
        #x, _ = pad_packed_sequence(x, batch_first=True)

        if not self.infer:
            x, tags, masks, classes = x
        
        x = self.transform(x) # needs dropout after transforming 
        #x = self.dropout(x) # dropout
        out, h = self.qrnn(x) #'out' has dimension(batchsize, seq_len, 2*hidden_size)
        #out = out.permute(1, 0, 2)
        out = self.linear(out) # now 'out' has dimension(batchsize, seq_len, num_class)
        # maybe add relu here
        #out = out.view(-1, out.shape[2]) # shape (128*seqlen, 18)
        #out = F.log_softmax(out, dim=1) # take the softmax across the dimension num_class, 'out' has dimension(batchsize, seq_len, num_class)
        out = out.view(out.shape[0], out.shape[2], out.shape[1])
        
        if not self.infer:
            outputs = out
            active_loss = masks.view(-1) == 1
            outputs = outputs.view(-1, len(classes))[active_loss]
            tags = tags.view(-1)[active_loss]
            print(outputs.shape)
            new_tags = []
            new_outputs = []
            for t, o in zip(tags, outputs):
                if int(t) != 0:
                    new_tags.append(t)
                    new_outputs.append(o)
                else:
                    new_tags.extend(torch.LongTensor([0, 7, 8]).to('cuda:0'))
                    new_outputs.extend([o]*3)
                    
            new_tags = [t.unsqueeze(0) for t in new_tags]
            new_outputs = [o.unsqueeze(0) for o in new_outputs]
            tags = torch.Tensor(len(new_outputs)).to('cuda:0')
            tags = torch.cat(new_tags)
            outputs = torch.Tensor(len(new_outputs), len(outputs[0])).to('cuda:0')
            outputs = torch.cat(new_outputs)
            print(tags.shape)
            
            print(outputs.shape)
            import sys
            sys.exit()

        return out
