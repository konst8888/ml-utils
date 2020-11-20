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
import argparse
from sklearn.metrics import confusion_matrix
from collections import Counter

sys.path.append(os.path.join('vidaug', 'vidaug'))
#os.chdir(os.path.join(os.path.abspath(os.path.curdir), 'vidaug', 'vidaug', 'augmentors'))
#from vidaug import augmentors as va
#import augmentors as va

from model import MultiResolution, VideoNet
from mobilenet_v2 import MobileNetV2
from dataset import MultiResDataset, FramesDataset
from utils import (
    get_split,
    load_state_dict
)
from transforms_video import (
    ScaleVideo,
    ResizeVideo
)


def adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer):
    if (sample_counter + 1) % adjust_lr_every >= 0 \
    and (sample_counter + 1) % adjust_lr_every < batch_size:  # 500
        for param in optimizer.param_groups:
            param['lr'] = max(param['lr'] / 1.2, 1e-4)


def train(
    model,
    criterion,
    optimizer,
    dataloader_train,
    dataloader_test,
    epochs,
    checkpoint_path,
    device,
    save_at,
    adjust_lr_every,
    use_multires):
    data_len = len(dataloader_test)
    batch_size = dataloader_train.batch_size
    sample_counter = 0
    if adjust_lr_every <= 1:
        adjust_lr_every = adjust_lr_every * data_len * batch_size
    adjust_lr_every = int(adjust_lr_every)
    for epoch in range(epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                dataloader = dataloader_train
                model.train()
            if phase == 'test':
                y_true = []
                y_pred = []
                idxs_fp = []
                idxs_fn = []
                dataloader = dataloader_test
                model.eval()
            running_loss = 0
            running_acc = 0
            pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
            for idx, sample in pbar:
                if phase == 'train':
                    sample_counter += batch_size
                    adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer)
                    
                #clips_context, clips_fovea, labels = sample
                idxs, clips, labels = sample
                clips = clips.to(device)
                labels = labels.to(device)

                if use_multires:
                    clips_fovea = clips_fovea.to(device)
                    inputs = [clips_context, clips_fovea]
                else:
                    inputs = clips
                outputs = model(inputs)
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
                    "Epoch: {}/{} Phase: {} Loss: {:.4f} Acc: {:.4f}".format(
                        epoch,
                        epochs - 1,
                        phase,
                        running_loss * scale_value,
                        running_acc * scale_value
                    )
                )
                if phase == 'train':
                    train_acc = running_acc * scale_value
                    train_loss = running_loss * scale_value
                if phase == 'test':
                    y_true.extend(labels.cpu().numpy().tolist())
                    y_pred.extend(preds.cpu().numpy().tolist())
                    idxs_fp.extend(idxs[(labels != preds) & (preds == 1)].cpu().numpy().tolist())
                    idxs_fn.extend(idxs[(labels != preds) & (preds == 0)].cpu().numpy().tolist())
                    if idx == data_len-1:
                        print(Counter(y_true))
                        print(confusion_matrix(y_true, y_pred))
                        if epoch >= 5:
                            with open('fp.txt', 'w') as file:
                                [file.write(f'{f}/n') for f in idxs_fp]
                            with open('fn.txt', 'w') as file:
                                [file.write(f'{f}/n') for f in idxs_fn]
               
                if phase == 'test' and checkpoint_path is not None \
                and idx in (int(data_len * save_at) - 1, data_len - 1):
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_path, 'non_personal_video_epoch_{}_acc_{:.4f}_vs_{:.4f}_loss_{:.4f}_vs_{:.4f}.pth'.format(
                            epoch,
                            train_acc,
                            running_acc * scale_value,
                            train_loss,
                            running_loss * scale_value
                            )))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data",
                        help="Path to data root dir")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--model_path", default='',
                        help="Load existing model path")
    parser.add_argument("--csv_path", default='',
                        help="csv file path")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_at", type=float, default=1,
                        help="Save checkpoint at current training stage, float in (0, 1)")
    parser.add_argument("--adjust_lr_every", type=float, default=1,
                        help="Lr decrease factor")
    parser.add_argument("--resize", type=int, default=180,
                        help="Resize")
    parser.add_argument("--time_depth", type=int, default=14,
                        help="Count of frames in video clip")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Count of workers")
    parser.add_argument("--use_multires", action='store_true',
                        help="Use Multiresolution net")


    args = parser.parse_args()

    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    save_at = args.save_at
    adjust_lr_every = args.adjust_lr_every
    time_depth = args.time_depth
    FRAME_SIZE = (args.resize, args.resize)
    use_multires = args.use_multires
    csv_path = args.csv_path
    num_workers = args.num_workers
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataloader = DataLoader(FlyingChairsDataset("../FlyingChairs2/"),
    # batch_size=1)
    transform = T.Compose([
        ScaleVideo(),
        #ResizeVideo(size=FRAME_SIZE)
    ])
    
    #sometimes = lambda aug: va.Sometimes(0.5, aug) # Used to apply augmentor with 50% probability
    #seq = va.Sequential([
    #    sometimes(va.RandomRotate(degrees=10)),
    #    sometimes(va.HorizontalFlip()) # horizontally flip the video with 50% probability
    #])

    train_index, test_index = get_split(data_path, csv_path, test_size=0.2)
    dataset_train = FramesDataset(
        root_dir=data_path, 
        csv_path=csv_path,
        time_depth=time_depth, 
        transform=transform,
        idxs=train_index
    )
    dataset_test = FramesDataset(
        root_dir=data_path, 
        csv_path=csv_path,
        time_depth=time_depth, 
        transform=transform,
        idxs=test_index
    )
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

    #for d in dataloader_train:
    #    clip_context, clip_fovea, label = d
    #    print(clip_context.shape, clip_context.shape)

    if use_multires:
        Net = MultiResolution
    else:
        #Net = VideoNet
        Net = MobileNetV2
        
    #model = Net(
    #    t_dim=time_depth, 
    #    img_x=FRAME_SIZE[0] // 2, 
    #    img_y=FRAME_SIZE[1] // 2
    #).to(device)
    
    model = MobileNetV2(num_classes=2, width_mult=1.0).to(device)
    
    print(sum([p.numel() for p in model.parameters()]) / 1e6)
    
    if model_path:
        model = load_state_dict(model, model_path, device)
        
        #model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train(
        model,
        criterion,
        optimizer,
        dataloader_train,
        dataloader_test,
        epochs,
        checkpoint_path,
        device,
        save_at,
         adjust_lr_every,
        use_multires
    )
  
    