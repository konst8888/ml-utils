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
    ELRLoss,
    AttnBiLSTM,
)
from dataset import (
    NERCharDataset,
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
    lam,
    wandb):
    #data_len = len(dataloader_test)
    data_len = 10
    sample_counter = 0
    print_idx = 1
    
    for epoch in range(start_epoch, epochs):
        print('LR: ', optimizer.param_groups[0]['lr'])
        for phase in ['train', 'valid']:
            if phase == 'train':
                dataloader = dataloader_train
                model.train()
            if phase == 'valid':
                dataloader = dataloader_test
                model.eval()
                
            batch_size = dataloader.batch_size
            if adjust_lr_every <= 1:
                adjust_lr_every = adjust_lr_every * data_len * batch_size
            adjust_lr_every = int(adjust_lr_every)
            running_loss = 0
            running_acc = 0
            running_class_acc = {cl: 0 for cl in classes}
            counts_rec = {cl: 0 for cl in classes}
            counts_prec = {cl: 0 for cl in classes}
            
            #criterion.set_phase(phase)
            pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
            current_pos = 0
            for idx, sample in pbar:
                seqs, labels, idxs, lengths = sample[0], sample[1], sample[2], sample[3]
                seqs = seqs.to(device)
                labels = labels.to(device)
                outputs = model(seqs, lengths)
                                  
                #print(outputs.shape)
                #print(labels.shape)
                labels = labels.view(-1)
                current_pos += len(labels)
                loss = criterion(outputs, labels)
                #loss, elr_loss = criterion(list(range(current_pos-len(labels), current_pos)), outputs, labels)
                
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
                                
                scale_value = 1 / batch_size / (idx + 1)
                running_loss += loss.item() * batch_size
                running_acc += (preds == labels.data).float().sum()
                for i, cl in enumerate(classes):
                    running_class_acc[cl] += ((preds == i) & (labels.data == i)).float().sum()
                    counts_rec[cl] += (labels.data == i).float().sum()
                    counts_prec[cl] += (preds == i).float().sum()
                
                pbar.set_description(
                    "Epoch: {}/{} Phase: {} Loss: {:.4f} ({:.4f}) Acc: {:.4f} ({:.4f}) Prec {}: {:.4f} ({:.4f})".format(
                        epoch,
                        epochs - 1,
                        phase,
                        running_loss * scale_value,
                        loss.item(),
                        running_acc * scale_value,
                        (preds == labels.data).float().mean(),
                        classes[print_idx],
                        running_class_acc[classes[print_idx]] / counts_prec[classes[print_idx]],
                        ((preds == labels.data) & (labels.data == print_idx)).float().sum() / (labels.data == print_idx).float().sum()
                    )
                )
                if phase == 'train':
                    train_acc = running_acc * scale_value
                    train_loss = running_loss * scale_value
            
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
            
            metrics.update({f'{k}_valid_rec': v for k, v in counts_rec.items()})
            metrics.update({f'{k}_valid_prec': v for k, v in counts_prec.items()})
            wandb.log(metrics)

        if phase == 'valid' and checkpoint_path is not None:
            torch.save(
                model.state_dict(),
                os.path.join(checkpoint_path, 'toxic_epoch_{}_acc_{:.4f}_vs_{:.4f}_loss_{:.4f}_vs_{:.4f}.pth'.format(
                    epoch,
                    train_acc,
                    running_acc * scale_value,
                    train_loss,
                    running_loss * scale_value
                    )))
            
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
        wandb.init(config=vars(args), project="nlp-toxic-deep") # resume=True
        wandb.save('*.py')


    seed = 8
    set_seed(seed)    
    test_size = 0.2
    
    #tp = TextProcessing()
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
        
    classes = [0, 1]
    
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        #collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=20,
        num_workers=num_workers,
        #collate_fn=collate_fn,
        shuffle=True,
        pin_memory=True,
    )
    
    model = AttnBiLSTM( #BiQRNN
        embedding_size=300, 
        hidden_size=100, 
        total_words=len(dataset_train.vocab), 
        num_class=len(dataset_train.class_proportion),
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
    
    class_proportion = dataset_train.class_proportion
    print(class_proportion)
    weight = (1. / torch.FloatTensor(class_proportion)).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=weight,
    )
    criterion1 = ELRLoss(
        num_examp={'train': int(1e7), 'valid': int(1e7)},
        #num_examp={'train': batch_size, 'valid': batch_size},
        num_classes=len(classes),
        beta=0.3, # 0.3
        lam=0.15,
        weight=weight,
        device=device,
        fix=True,
    )
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
        args.lam,
         wandb,
    )
  
    