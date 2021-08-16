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
import numpy as np
import pandas as pd
import argparse
import random
from sklearn.metrics import confusion_matrix
from collections import Counter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import AudioCNN
from dataset import AudioDataset
from utils import (
    get_split,
    load_state_dict
)

def adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer):
    if (sample_counter + 1) % adjust_lr_every >= 0 \
    and (sample_counter + 1) % adjust_lr_every < batch_size:  # 500
        for param in optimizer.param_groups:
            param['lr'] = max(param['lr'] / 1.2, 1e-4)

def set_seed(seed=8):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def train(
    model,
    criterion,
    optimizer,
    scheduler,
    dataloader_train,
    dataloader_test,
    epochs,
    start_epoch,
    checkpoint_path,
    device,
    save_at,
    adjust_lr_every,
    wandb):
    data_len = len(dataloader_test)
    batch_size = dataloader_train.batch_size
    sample_counter = 0
    if adjust_lr_every <= 1:
        adjust_lr_every = adjust_lr_every * data_len * batch_size
    adjust_lr_every = int(adjust_lr_every)
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
            pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
            for idx, sample in pbar:
                if phase == 'train':
                    sample_counter += batch_size
                    #adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer) # Plateau
                    
                #clips_context, clips_fovea, labels = sample
                imgs, labels = sample
                #print(imgs.shape)
                #print('labels', labels)
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = criterion(outputs, labels)
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                scale_value = 1 / batch_size / (idx + 1)
                running_loss += loss.item() * batch_size
                running_acc += (preds == labels.data).float().sum()
                pbar.set_description(
                    "Epoch: {}/{} Phase: {} Loss: {:.4f} ({:.4f}) Acc: {:.4f} ({:.4f})".format(
                        epoch,
                        epochs - 1,
                        phase,
                        running_loss * scale_value,
                        loss.item(),
                        running_acc * scale_value,
                        (preds == labels.data).float().mean()
                    )
                )
                if phase == 'train':
                    train_acc = running_acc * scale_value
                    train_loss = running_loss * scale_value
               
                if phase == 'valid' and checkpoint_path is not None \
                        and idx in (int(data_len * save_at) - 1, data_len - 1):
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_path, 'edm_genre_epoch_{}_acc_{:.4f}_vs_{:.4f}_loss_{:.4f}_vs_{:.4f}.pth'.format(
                            epoch,
                            train_acc,
                            running_acc * scale_value,
                            train_loss,
                            running_loss * scale_value
                            )))

        scheduler.step(running_loss * scale_value)

        if wandb is not None:
            metrics = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': running_loss * scale_value,
                'valid_acc': running_acc * scale_value,
            }
            wandb.log(metrics)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data",
                        help="Path to data root dir")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--model_path", default='',
                        help="Load existing model path")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Start epoch num")
    parser.add_argument('--use-wandb', help='Whether use wandb', action='store_true')
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_at", type=float, default=1,
                        help="Save checkpoint at current training stage, float in (0, 1)")
    parser.add_argument("--adjust_lr_every", type=float, default=1,
                        help="Lr decrease factor")
    parser.add_argument("--resize", type=int, default=180,
                        help="Resize")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Count of workers")

    args = parser.parse_args()

    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs
    start_epoch = args.start_epoch
    use_wandb = args.use_wandb
    lr = args.lr
    save_at = args.save_at
    adjust_lr_every = args.adjust_lr_every
    FRAME_SIZE = (args.resize, args.resize)
    num_workers = args.num_workers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    wandb = None
    if use_wandb:
        import wandb
        wandb.login(key='8bbaf831241b8525faf8ba0890cc99d295161186')
        wandb.init(config=vars(args), project="edm-genre-classify")


    seed = 8
    set_seed(seed)
    transform = T.Compose([
        #T.ToTensor(),
        T.Lambda(lambda x: torch.FloatTensor(x / 255.0 * 2 - 1)),
    ])
    
    data = np.load(data_path, mmap_mode='r', allow_pickle=True)
    labels = data['y']
    print('Class proportion: ')
    print(pd.Series(labels).value_counts(normalize=True))
    classes = list(set(labels))
    num_classes = len(classes)

    train_index, valid_index = get_split(labels, num_classes, test_size=0.1, seed=seed)
    with open('valid_index.json', 'w') as file:
        for idx in valid_index:
            file.write(str(idx) + '\n')

    if not os.path.exists('class_map.json'):
        with open('class_map.json', 'w') as file:
            for cl in classes:
                file.write(cl + '\n')
    else:
        with open('class_map.json', 'r') as file:
            classes = file.read().split('\n')[:-1]

    labels = [classes.index(l) for l in labels]
    dataset_train = AudioDataset(
        data, train_index, classes, transform
    )
    dataset_test = AudioDataset(
        data, valid_index, classes, transform
    )
    del data
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    
    model = AudioCNN(num_classes=num_classes).to(device)
    
    print(sum([p.numel() for p in model.parameters()]) / 1e6)
    
    if model_path:
        model = load_state_dict(model, model_path, device)
        
        #model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.04)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.3)
    criterion = nn.CrossEntropyLoss()
    train(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloader_train,
        dataloader_test,
        epochs,
        start_epoch,
        checkpoint_path,
        device,
        save_at,
         adjust_lr_every,
         wandb,
    )
  
    