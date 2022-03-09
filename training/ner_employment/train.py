import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
from PIL import Image
import tqdm
import sys
import os
import re
import numpy as np
import pandas as pd
import yaml
import argparse
import random
from sklearn.metrics import confusion_matrix
from collections import Counter, defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau
from warmup_scheduler import GradualWarmupScheduler
from model import (
    BiLSTM,
    BiQRNN,
    NoiseBiLSTM,
    ELRLoss
)
from dataset import (
    #NERDataset,
    NERProjDataset,
    NERCharDataset,
    TextProcessing,
)
from utils import (
    get_split,
    load_state_dict,
    collate_fn,
)

def set_seed(seed=8):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def train(
    model,
    criterion,
    optimizer,
    scheduler,
    dataset_train,
    dataloader_train,
    dataloader_test,
    epochs,
    start_epoch,
    checkpoint_path,
    device,
    save_at,
    adjust_lr_every,
    classes,
    max_seq_len,
    lam,
    wandb):
    #data_len = len(dataloader_test)
    data_len = 10
    batch_size = dataloader_train.batch_size
    sample_counter = 0
    if adjust_lr_every <= 1:
        adjust_lr_every = adjust_lr_every * data_len * batch_size
    adjust_lr_every = int(adjust_lr_every)
    
    scores_empl = []
    tokens_empl = []
    #running_scores = defaultdict(lambda x: defaultdict(0.))
    running_scores = defaultdict(lambda: defaultdict(list))
    relabel = []
    use_relabel = False
    all_tokens = []
    change_idxs = [4]
    print_idx = 5
    relabel_every = 5
    thresh = 0.95
    
    for epoch in range(start_epoch, epochs):
        print('LR: ', optimizer.param_groups[0]['lr'])
        for phase in ['train', 'valid']:
            if phase == 'train':
                dataloader = dataloader_train
                model.train()
            if phase == 'valid':
                dataloader = dataloader_test
                model.eval()
            running_loss = 0
            running_acc = 0
            running_elr_loss = 0
            running_class_acc = {cl: 0 for cl in classes}
            counts_rec = {cl: 0 for cl in classes}
            counts_prec = {cl: 0 for cl in classes}
            
            criterion.set_phase(phase)
            pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
            current_pos = 0
            for idx, sample in pbar:
                seqs, masks, tags, tokens, idxs = sample[0], sample[1], sample[2], sample[3], sample[4]
                seqs = seqs.to(device)
                masks = masks.to(device)
                tags = tags.to(device)
                outputs = model(seqs)
                
                active_loss = masks.view(-1) == 1
                outputs = outputs.view(-1, len(classes))[active_loss]
                tags = tags.view(-1)[active_loss]
                current_pos += len(tags)
                
                tokens = np.array(tokens)
                tokens = tokens.reshape(np.prod(tokens.shape))[active_loss.tolist()]
                
                if use_relabel and phase == 'train':
                    if epoch == start_epoch:
                        relabel.extend(tags.tolist())
                        all_tokens.extend(tokens[: len(tags)])
                    else:
                        tags = relabel[current_pos: current_pos + len(tags)]
                        tags = torch.LongTensor(tags).to(device)
                        #current_pos += len(tags)
                    for i in change_idxs:
                        running_scores[i][epoch].extend(
                            (outputs.exp() / outputs.exp().sum(dim=1, keepdim=True))[:, i].view(-1).tolist()
                        )
                
                #loss = criterion(outputs, tags)
                loss, elr_loss = criterion(list(range(current_pos-len(tags), current_pos)), outputs, tags)

                """
                l2_lambda = 0.006 # 3
                l2_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l2_reg += torch.norm(param)
                loss += l2_lambda * l2_reg
                """
                
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                
                #scores_empl.extend((outputs.exp() / outputs.exp().sum(dim=1, keepdim=True))[:, print_idx].view(-1).tolist())
                #tokens_empl.extend(tokens.tolist())
                if idx == len(dataloader) - 1 and phase == 'train':
                    if epoch == start_epoch and idx == 0:
                        with open('words_of_interest.txt', 'w') as f:
                            f.write('')
                    with open('words_of_interest.txt', 'a+') as f:
                        f.write(str(epoch) + '\n')
                        f.write(str(sorted(zip(tokens_empl, scores_empl), key=lambda x: x[1], reverse=True)[::100][:100])+'\n')
                
                scale_value = 1 / batch_size / max_seq_len / (idx + 1)
                running_loss += loss.item() * batch_size * max_seq_len
                running_acc += (preds == tags.data).float().sum()
                running_elr_loss += elr_loss
                for i, cl in enumerate(classes):
                    running_class_acc[cl] += ((preds == tags.data) & (tags.data == i)).float().sum()
                    counts_rec[cl] += (tags.data == i).float().sum()
                    counts_prec[cl] += (preds == tags.data).float().sum()
                
                pbar.set_description(
                    "Epoch: {}/{} Phase: {} Loss: {:.4f} ({:.4f}) Acc: {:.4f} ({:.4f}) Rec {}: {:.4f} ({:.4f})".format(
                        epoch,
                        epochs - 1,
                        phase,
                        running_loss * scale_value,
                        loss.item(),
                        running_acc * scale_value,
                        (preds == tags.data).float().mean(),
                        classes[print_idx],
                        running_class_acc[classes[print_idx]] / counts_rec[classes[print_idx]],
                        ((preds == tags.data) & (tags.data == print_idx)).float().sum() / (tags.data == print_idx).float().sum()
                    )
                )
                if phase == 'train':
                    train_acc = running_acc * scale_value
                    train_loss = running_loss * scale_value
                    train_elr_loss = running_elr_loss * scale_value
                           
        if use_relabel and epoch > 0 and epoch % relabel_every == 0:
        #if epoch == 0:
            inst_count = len(relabel)
            running_scores = {i: [np.mean([ss[j] for ep, ss in running_scores[i].items()]) for j in range(inst_count)] for i in running_scores}
            
            assert all(len(all_tokens) == len(running_scores[i]) for i in running_scores), f'{len(all_tokens)} vs {len(running_scores[8])}'
            
            all_text = ''.join(all_tokens)
            words = NERProjDataset.tokenize(all_text)
            
            idxs_affected = set()
            idxs_appended = defaultdict(list)
            """
            for i in running_scores:
                for j, score in enumerate(running_scores[i]):                    
                    if relabel[j] == 0 and score > thresh and j not in idxs_affected:
                        if i == 8 and (j > 0 and relabel[j-1] not in {7, 8} or j == 0):
                            continue
                        relabel[j] = i
                        idxs_affected.add(j)
                        idxs_appended[i].append(j)
            """
            
            for i in running_scores:
                for j, (score, token) in enumerate(zip(running_scores[i], all_tokens)):                    
                    if relabel[j] == 0 and score > thresh and j not in idxs_affected:
                        if token in (' ', ',', '.', '!', '?', ';'):
                            continue
                        #if j < len(all_tokens) and all_tokens[j+1] in (' ', ',', '.'):
                        #    continue
                        relabel[j] = i
                        for k in range(3):
                            if i-k < 0 or i+k > len(all_tokens)-1 or all_tokens[i+k] in (' ', '\n') or all_tokens[i-k] in (' ', '\n'):
                                break
                            relabel[j+k] = i
                            relabel[j-k] = i
                            
                        idxs_affected.add(j)
                        idxs_appended[i].append(j)

            running_scores = defaultdict(lambda: defaultdict(list))
            print({f'{classes[k]}_appended_data': len(v) for k, v in idxs_appended.items()})
            #print({i: np.array(all_tokens)[idxs_appended[i][:30]] for i in idxs_appended})
            #print({i: idxs_appended[i][:30] for i in idxs_appended})
            #all_tokens = []
            
        if wandb is not None:
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': running_loss * scale_value,
                'valid_acc': running_acc * scale_value,
                'LR': optimizer.param_groups[0]['lr'],
            }
            for i, cl in enumerate(classes):
                counts_rec[cl] = running_class_acc[cl] / counts_rec[cl]
                counts_prec[cl] = running_class_acc[cl] / counts_prec[cl]
            
            metrics.update({k + '_valid_rec': v for k, v in counts_rec.items()})
            metrics.update({k + '_valid_prec': v for k, v in counts_prec.items()})
            if isinstance(criterion, ELRLoss):
                metrics.update({
                    'elr_loss_train': train_elr_loss,
                    'elr_loss_valid': running_elr_loss * scale_value,
                })
            if use_relabel and epoch > 0 and epoch % relabel_every == 0:
                metrics.update({f'{classes[k]}_appended_data': len(v) for k, v in idxs_appended.items()})
            wandb.log(metrics)

        if phase == 'valid' and checkpoint_path is not None:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, 'ner_base_epoch_{}_acc_{:.4f}_vs_{:.4f}_loss_{:.4f}_vs_{:.4f}.pth'.format(
                    epoch,
                    train_acc,
                    running_acc * scale_value,
                    train_loss,
                    running_loss * scale_value
                    )))
            
        if False and optimizer.param_groups[0]['lr'] <= 5e-4:
            class_proportion = dataset_train.class_proportion
            print(class_proportion)
            weight = torch.FloatTensor([1/sum(class_proportion)/10] + [1 / cp for cp in class_proportion]).to(device)
            criterion = nn.CrossEntropyLoss(weight = weight)

        if epoch >= 20:
            class_proportion = dataset_train.class_proportion[:]
            class_proportion[classes[1:].index('AGE')] *= 10
            class_proportion[classes[1:].index('PERSON_FAM')] /= 2
            print(class_proportion)
            weight = torch.FloatTensor([1/class_proportion[classes[1:].index('GPE')]/60] + [1 / cp for cp in class_proportion]).to(device) # /30
            criterion = ELRLoss(
                num_examp={'train': int(1e7), 'valid': int(1e7)},
                #num_examp={'train': batch_size, 'valid': batch_size},
                num_classes=len(dataset_train.tag2index),
                beta=0.3, # 0.5 --best
                lam=1.5,
                weight=weight,
                device=device,
                fix=False
            )

            
        scheduler.step(running_loss * scale_value)

        #local_vars = list(locals().items()) + list(globals().items())
        #for var, obj in local_vars:
        #    print(var, sys.getsizeof(obj))

        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data",
                        help="Path to data root dir")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--model_path", default='',
                        help="Load existing model path")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Start epoch num")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test size")
    parser.add_argument("--lam", default=0, type=float, help="Regularization coef")
    parser.add_argument("--patience", default=2, type=int, help="Patience of sheduler")
    parser.add_argument("--factor", default=0.5, type=float, help="LR decrease factor")
    parser.add_argument('--use-wandb', help='Whether use wandb', action='store_true')
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_at", type=float, default=1,
                        help="Save checkpoint at current training stage, float in (0, 1)")
    parser.add_argument("--adjust_lr_every", type=float, default=1,
                        help="Lr decrease factor")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Count of workers")

    args = parser.parse_args()

    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs
    start_epoch = args.start_epoch
    test_size = args.test_size
    use_wandb = args.use_wandb
    lr = args.lr
    save_at = args.save_at
    adjust_lr_every = args.adjust_lr_every
    num_workers = args.num_workers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    print(device)

    wandb = None
    if use_wandb:
        import wandb
        wandb.login(key='8bbaf831241b8525faf8ba0890cc99d295161186')
        wandb.init(config=vars(args), project="nlp-ner-employment")
        wandb.save('*.py')


    seed = 8
    set_seed(seed)    
    test_size = 0.2
    
    tp = TextProcessing()
    transforms = (
        lambda x: x.lower() 
        if random.random() < 0.3
        else x
    )
    dataset_train = NERCharDataset(
        data_path, 
        'train', 
        transforms=transforms, 
        test_size=test_size
    )
    
    dataset_test = NERCharDataset(
        data_path, 
        'test', 
        transforms=transforms, 
        test_size=test_size,
        add_data = {
        #    'length': dataset_train.length,
            'max_seq_len': dataset_train.max_seq_len,
            'vocab': dataset_train.vocab,
        }
    )
        
    classes = list(dataset_train.tag2index.keys())
    #max_seq_len = dataset_train.max_seq_len
    max_seq_len = 128 #256
    
    vocab = dataset_train.vocab
    """
    with open('char_embeddings.json', 'r') as f:
        fixed_embeds = yaml.safe_load(f.read())
    
    pattern = re.compile("[a-zA-Zа-яА-Я]+")
    fixed_embeds = {
        k: v for k, v in fixed_embeds.items() 
            if bool(pattern.fullmatch(k))
    }
    for i, (k, v) in enumerate(fixed_embeds.items()):
        vocab[k.encode('unicode-escape').decode('ASCII')] = i
    
    for k, v in vocab.items():
        if not (k.encode('ASCII').decode('unicode-escape') in fixed_embeds):
            i += 1
            vocab[k] = i
    
    dataset_train.vocab = vocab
    dataset_test.vocab = vocab
    """
    
    #with open(f'char_vocab.json', 'w', encoding='unicode-escape') as f:
    #    f.write(str(vocab))
    
    #assert set(vocab.values()) == set(list(range(len(vocab))))
    #print(len(dataset_train.vocab))
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        #collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=20,
        num_workers=num_workers,
        #collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )
    
    model = BiLSTM( #BiQRNN
        embedding_size=300, 
        hidden_size=100, 
        total_words=len(dataset_train.vocab), 
        num_class=len(dataset_train.tag2index),
        fixed_embeds=None, # fixed_embeds
        #device=device
    ).to(device)
    
    #print(model(torch.randn(1, 2, 100, 100).to(device)).shape)
    
    print(sum([p.numel() for p in model.parameters()]) / 1e6)
    if model_path:
        model_path = os.path.join(checkpoint_path, model_path)
        model = load_state_dict(model, model_path, device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    """
    optimizer = optim.Adam(
        [
            {"params": model.wordembed.parameters(), "lr": lr / 3},
            {"params": model.bilstm.parameters(), "lr": lr},
            {"params": model.linear.parameters(), "lr": lr},
        ],
        lr=lr,
    )
    """
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=args.factor, min_lr=1e-5)
    #scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=3, after_scheduler=scheduler_pl)
    
    class_proportion = dataset_train.class_proportion[:]
    class_proportion[classes[1:].index('AGE')] = np.inf
    class_proportion[classes[1:].index('EMPLOYMENT')] = np.inf
    #class_proportion[classes[1:].index('AGE')] *= 10
    print(class_proportion)
    #weight = torch.FloatTensor([0] + [1 / cp for cp in class_proportion]).to(device)
    weight = torch.FloatTensor([1/class_proportion[classes[1:].index('GPE')]/60] + [1 / cp for cp in class_proportion]).to(device) # /30
    #weight = torch.FloatTensor([0]*7 + [1 / cp for cp in class_proportion[6:8]]).to(device)
    criterion = nn.CrossEntropyLoss(
        #ignore_index=0,
        #weight=torch.FloatTensor([0.]*7 + [1]*2).to(device),
        weight=weight,
    )
    #"""
    criterion = ELRLoss(
        num_examp={'train': int(1e7), 'valid': int(1e7)},
        #num_examp={'train': batch_size, 'valid': batch_size},
        num_classes=len(dataset_train.tag2index),
        beta=0.3, # 0.3
        lam=1.5,
        weight=weight,
        device=device,
        fix=True,
    )
    #"""
    train(
        model,
        criterion,
        optimizer,
        scheduler,
        dataset_train,
        dataloader_train,
        dataloader_test,
        epochs,
        start_epoch,
        checkpoint_path,
        device,
        save_at,
         adjust_lr_every,
        classes,
        max_seq_len,
        args.lam,
         wandb,
    )
  
    