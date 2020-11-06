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
import argparse

from model import MultiResolution
from dataset import MultiResDataset
from utils import get_split
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
     adjust_lr_every):
    data_len = len(dataloader_train)
    batch_size = dataloader_train.batch_size
    sample_counter = 0
    if adjust_lr_every < 1:
        adjust_lr_every = adjust_lr_every * data_len * batch_size
    adjust_lr_every = int(adjust_lr_every)
    for epoch in range(epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                dataloader = dataloader_train
                model.train()
            if phase == 'test':
                dataloader = dataloader_test
                model.eval()
            running_loss = 0
            running_acc = 0
            pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
            for idx, sample in pbar:
                for s in sample:
                    s.to(device)
                #clips_context = clips_context.to(device)
                clips_context, clips_fovea, labels = sample


                if phase == 'train':
                    sample_counter += batch_size
                    adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer)

                outputs = model([clips_context, clips_fovea])
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                scale_value = 1 / batch_size / max(idx, 1)
                running_loss += loss.item()
                pbar.set_description(
                    "Epoch: {}/{} Loss: {:.4f}".format(
                        epoch,
                        epochs,
                        running_loss * scale_value,
                    )
                )
                if checkpoint_path is not None and idx in (int(data_len * save_at) - 1, data_len - 1):
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_path, 'reconet_phase_{}_epoch_{}_loss_{:.4f}.pth'.format(
                            phase,
                            epoch,
                            loss)))


if __name__ == '__main__':
    # python3 train.py --data_path /home/konstantinlipkin/Anaconda_files/data_test --style_path /home/konstantinlipkin/Anaconda_files/data_path/some_class/image.jpg --phase first
    
    # python3 train.py --data_path ~/konst/data/moderation_resized --style_path ~/konst/ml-utils/training/neural_style_transfer/styles --checkpoint_path ~/konst/model_checkpoints/neural_style_transfer --phase first --batch_size 12 --manual_weights --alpha 5e5 --beta 1e3 --gamma 1e-6 --epochs 2 --save_at 0.5 --adjust_lr_every 0.3
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data",
                        help="Path to data root dir")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--model_path", default='',
                        help="Load existing model path")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_at", type=float, default=1,
                        help="Save checkpoint at current training stage, float in (0, 1)")
    parser.add_argument("--adjust_lr_every", type=float, default=1,
                        help="Lr decrease factor")
    parser.add_argument("--time_depth", type=float, default=16,
                        help="Count of frames in video clip")

    args = parser.parse_args()

    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    save_at = args.save_at
    adjust_lr_every = args.adjust_lr_every
    time_depth = int(args.time_depth)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataloader = DataLoader(FlyingChairsDataset("../FlyingChairs2/"),
    # batch_size=1)
    FRAME_SIZE = (180, 180) # 256, 256
    transform = T.Compose([
        ScaleVideo(),
        ResizeVideo(size=FRAME_SIZE)
    ])

    train_index, test_index = get_split(data_path)
    dataset_train = MultiResDataset(
        root_dir=data_path, 
        time_depth=time_depth, 
        transform=transform,
        idxs=train_index
    )
    dataset_test = MultiResDataset(
        root_dir=data_path, 
        time_depth=time_depth, 
        transform=transform,
        idxs=test_index
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        shuffle=True
    )

    for d in dataloader_train:
        clip_context, clip_fovea, label = d
        print(clip_context.shape, clip_context.shape)


    model = MultiResolution(
        t_dim=time_depth, 
        img_x=FRAME_SIZE[0] // 2, 
        img_y=FRAME_SIZE[1] // 2
    ).to(device)

    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    for p in model.parameters():
        p.requires_grad = True

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
         adjust_lr_every
    )
  
    