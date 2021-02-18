import torch
import torch.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import sys
import argparse
import random
import os
import numpy as np
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
import face_detection

from model import ReCoNetMobile
from utils import (
    additional_augmenation,
    load_state_dict,
    normalize,
    normalize_after_reconet
)
from network import ResNet18, Vgg16
from dataset import DatasetGenFace
#from frn import *

def calc_sim_weights(img, style):
    if not use_sim:
        l = len(style)
        return [1. / l] * l
    if len(style) == 1:
    	return [1.]
    	
    style_sim_features = [resnet(s) for s in style]
    img_sim_feature = resnet(img)
    #cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    weights = [img_sim_feature.sub(feature).pow(2).sum()
    		for feature in style_sim_features]
    weights = np.array(weights) / sum(weights)
    
    return weights

def calc_content_loss(feat1, feat2, alpha):
    out = L2distance(feat1[2], feat2[2].expand(feat1[2].shape))
    out *= alpha / (feat1[2].shape[1] * feat1[2].shape[2] * feat1[2].shape[3])
                 
    return out


def calc_l1_loss(tensor1, tensor2):
    out = gamma * tensor1.sub(tensor2).abs().sum()
    
    return out

def calc_l1_loss2(target, pred):
    
    class WeightDist:

        def __init__(self, size, x0, y0):
            self.x0 = np.ones(size) * x0
            self.y0 = np.ones(size) * y0
            self.sigma_x = 100
            self.sigma_y = 60
            self.C = 10

        def __call__(self, x, y):
            out = np.exp(-(((x - self.x0) / self.sigma_x)**2 + ((y - self.y0) / self.sigma_y)**2))
            out = 1 + self.C * out
            
            return out

    def get_weight_mask(size, x0, y0):
        wd = WeightDist(size, x0, y0)
        y, x = np.meshgrid(np.arange(size[0]), np.arange(size[1]), indexing='ij')
        res = wd(x, y)
        return res
    
    target_np = target[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    try:
        res = mtcnn.detect(target_np, landmarks=True)[2][0]
        assert len(res) > 0
    except:
        print('Face not found')
        return gamma * 1.5 * target.sub(pred).abs().sum()
    
    x1, y1 = [int(x) for x in res[3]]
    x2, y2 = [int(x) for x in res[4]]
    mask = get_weight_mask((600, 600), (x1 + x2) / 2, y1)
    
    out = target.sub(pred).abs()
    out *= torch.from_numpy(mask).to(device)
    out = gamma * out.sum()
    
    return out

def get_crop_bounds(pic):
    detection = face_detector.detect(pic)
    detection = detection[0][0:4]
    detection = [max(int(x), 0) for x in detection]
    return detection

def calc_vggface_loss(tensor1, tensor2):
    np1 = tensor1[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    np2 = tensor2[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    try:
        x1, y1, x2, y2 = get_crop_bounds(np1)
        face1 = np.array(np1)[y1:y2, x1:x2]
        face2 = np.array(np2)[y1:y2, x1:x2]
        face1 = torch.from_numpy(face1).div(255.0).mul(2).sub(1).permute(2, 0, 1).to(device)
        face2 = torch.from_numpy(face2).div(255.0).mul(2).sub(1).permute(2, 0, 1).to(device)
    except IndexError:
        print('Face not found')
        face1 = tensor1[0]
        face2 = tensor2[0]
        
    emb1 = vggface(face1.unsqueeze(0))
    emb2 = vggface(face2.unsqueeze(0))
    
    out = beta * L2distance(emb1, emb2)
    
    return out

def calc_perceptual_loss(target, pred, numpy=False):
    target_np = target[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    pred_np = pred[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    try:
        res = mtcnn.detect(target_np, landmarks=True)[2][0]
        assert len(res) > 0
    except:
        print('Face not found')
        return torch.cuda.FloatTensor([1.])
    
    x1, y1 = [int(x) for x in res[3]]
    x2, y2 = [int(x) for x in res[4]]
    bound_width = 10
    height = 100
    left_x = max(x1-bound_width, 0)
    right_x = min(x2+bound_width, target_np.shape[1])
    low_y = max(y1-height//6, 0)
    up_y = min(y1+height//2, target_np.shape[0])
    mouth_target = target_np[low_y:up_y, left_x:right_x, :]
    mouth_pred = pred_np[low_y:up_y, left_x:right_x, :]
    #mouth_target = normalize(torch.from_numpy(mouth_target).div(255.0).mul(2).sub(1).permute(2,0,1).to(device))
    #mouth_pred = normalize(torch.from_numpy(mouth_pred).div(255.0).mul(2).sub(1).permute(2,0,1).to(device))
    
    try:
        #out = calc_content_loss(
        #    Vgg16(mouth_target.unsqueeze(0)),
        #    Vgg16(mouth_pred.unsqueeze(0)),
        #    alpha
        #)
        if not numpy:
            out = mouth_target.sub(mouth_pred).abs().mean() # sum(); * beta
        else:
            out = float(np.mean(np.abs(mouth_target - mouth_pred)))
    except:
        #print('Align out of bounds')
        return torch.cuda.FloatTensor([1.])
        
    return out

def analyze_mouths(target, source, pred):
    target_np = target[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    pred_np = pred[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    source_np = source[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    try:
        res = mtcnn.detect(target_np, landmarks=True)[2][0]
        assert len(res) > 0
    except:
        print('Face not found')
        return torch.cuda.FloatTensor([1.])
    
    x1, y1 = [int(x) for x in res[3]]
    x2, y2 = [int(x) for x in res[4]]
    bound_width = 10
    height = 100
    left_x = max(x1-bound_width, 0)
    right_x = min(x2+bound_width, target_np.shape[1])
    low_y = max(y1-height//6, 0)
    up_y = min(y1+height//2, target_np.shape[0])
    mouth_target = target_np[low_y:up_y, left_x:right_x, :]
    mouth_pred = pred_np[low_y:up_y, left_x:right_x, :]
    score = float(np.mean(np.abs(mouth_target - mouth_pred)))
    try:
        Image.fromarray(mouth_target).save(f'../../../data/analyze_data/target_{round(score, 4)}.jpg')
        Image.fromarray(mouth_pred).save(f'../../../data/analyze_data/pred_{round(score, 4)}.jpg')
        Image.fromarray(source_np).save(f'../../../data/analyze_data/source_{round(score, 4)}.jpg')
    except Exception as e:
        print(e)
    
def reset_dataloader(ixs, is_seed=False):
    dataset_train.filter_index(ixs, is_seed)
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    return dataloader_train

def calc_mouth_no_teeth(target):
    target_np = target[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    try:
        res = mtcnn.detect(target_np, landmarks=True)[2][0]
        assert len(res) > 0
    except:
        print('Face not found')
        return torch.cuda.FloatTensor([10000.])
    
    x1, y1 = [int(x) for x in res[3]]
    x2, y2 = [int(x) for x in res[4]]
    bound_width = 10
    height = 100
    left_x = max(x1-bound_width, 0)
    right_x = min(x2+bound_width, target_np.shape[1])
    low_y = max(y1-height//6, 0)
    up_y = min(y1+height//2, target_np.shape[0])
    mouth_target = target_np[low_y:up_y, left_x:right_x, :]
    mouths_no_teeth = [Image.open(f'noTeethMouth{i}.jpg') for i in range(3)]
    mouths_teeth = [Image.open(f'TeethMouth{i}.jpg') for i in range(3)]
    
    max_shape_idx = np.argmax([np.prod(np.array(m).shape) for m in mouths_no_teeth] + [np.prod(np.array(m).shape) for m in mouths_teeth] + [np.prod(mouth_target.shape)])
    max_shape = [np.array(m).shape for m in mouths_no_teeth] + [np.array(m).shape for m in mouths_teeth] + [mouth_target.shape]
    max_shape = max_shape[max_shape_idx][:2][::-1]
    mouths_no_teeth = [m.resize(max_shape, Image.BICUBIC) for m in mouths_no_teeth]
    mouths_teeth = [m.resize(max_shape, Image.BICUBIC) for m in mouths_teeth]
    try:
        mouth_target = Image.fromarray(mouth_target).resize(max_shape, Image.BICUBIC)
    except ValueError as e:
        print(e)
        return 10000
    mouth_target = normalize(torch.from_numpy(np.array(mouth_target)).div(255.0).mul(2).sub(1).permute(2,0,1).to(device))
    mouths_no_teeth = [normalize(torch.from_numpy(np.array(m)).div(255.0).mul(2).sub(1).permute(2,0,1).to(device)) for m in mouths_no_teeth]
    mouths_teeth = [normalize(torch.from_numpy(np.array(m)).div(255.0).mul(2).sub(1).permute(2,0,1).to(device)) for m in mouths_teeth]
    
    #print(Vgg16(mouth_target.unsqueeze(0))[2].shape)
    #print(Vgg16(mouths[0].unsqueeze(0))[2].shape)
    out = np.mean([
        float(calc_content_loss(
            Vgg16(mouth_target.unsqueeze(0)),
            Vgg16(mouth.unsqueeze(0)),
            alpha)
        ) for mouth in mouths_no_teeth
    ])
    #"""
    out -= np.mean([
        float(calc_content_loss(
            Vgg16(mouth_target.unsqueeze(0)),
            Vgg16(mouth.unsqueeze(0)),
            alpha)
        ) for mouth in mouths_teeth
    ])
    #"""
    return out

def calc_mouth_no_teeth2(target):
    target_np = target[0].add(1.0).div(2).mul(255.0).permute(1, 2, 0).cpu().detach().numpy().astype('uint8')
    try:
        res = mtcnn.detect(target_np, landmarks=True)[2][0]
        assert len(res) > 0
    except:
        print('Face not found')
        return torch.cuda.FloatTensor([1.])
    
    x1, y1 = [int(x) for x in res[3]]
    x2, y2 = [int(x) for x in res[4]]
    bound_width = 10
    height = 100
    left_x = max(x1-bound_width, 0)
    right_x = min(x2+bound_width, target_np.shape[1])
    low_y = max(y1-height//6, 0)
    up_y = min(y1+height//2, target_np.shape[0])
    mouth_target = target_np[low_y:up_y, left_x:right_x, :]
    
    #out = np.sum(mouth_target >= 200)
    thresh = 100
    mask = (mouth_target[...,0]>=thresh) & (mouth_target[...,1]>=thresh) & (mouth_target[...,2]>=thresh)
    out = mask.sum()
    
    return out

        
def calc_reg_loss(styled_img, gamma):
    out = gamma * \
        (torch.sum(torch.abs(styled_img[:, :, :, :-1] - styled_img[:, :, :, 1:])) +
         torch.sum(torch.abs(styled_img[:, :, :-1, :] - styled_img[:, :, 1:, :])))
         
    return out
    
def adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer):
    if (sample_counter + 1) % adjust_lr_every >= 0 \
    and (sample_counter + 1) % adjust_lr_every < batch_size:  # 500
        for param in optimizer.param_groups:
            param['lr'] = max(param['lr'] / 1.2, 1e-4)
        print('lr: ', param['lr'])
        #print(sample_counter, adjust_lr_every)


def train(model, dataloader_train, dataloader_test, optimizer, L2distance, Vgg16, alpha, beta, gamma, epochs, checkpoint_path, device, save_at, adjust_lr_every):
    data_len = len(dataloader_train)
    batch_size = {
       'train': dataloader_train.batch_size,
        'valid': dataloader_test.batch_size
    }
    sample_counter = 0
    saving_points = [int(data_len * x * save_at) -
            1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1]
    print(saving_points)
    #if adjust_lr_every <= 10:
    #    adjust_lr_every = adjust_lr_every * data_len * batch_size['train']
    #adjust_lr_every = int(adjust_lr_every)
    for epoch in range(epochs):
        for phase in [
            'train', 
            'valid'
        ]:
            if phase == 'train':
                model.train()
                dataloader = dataloader_train
            else:
                model.eval()
                dataloader = dataloader_test

            running_content_loss = 0
            running_style_loss = 0
            running_reg_loss = 0
            running_vggface_loss = 0
            running_l1_loss = 0
            
            """
            # clean dataset from pairs (teeth, no_teeth)
            if phase == 'train' and epoch == 0:
                model.eval()
                thresh = 108
                best_ixs = []
                pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
                
                for idx, (ixs, sources, targets) in pbar:
                    
                    #if idx == 20:
                    #    break

                    for ix, source, target in zip(ixs, sources, targets):
                        #print(sample)
                        #source, target = sample
                        source = source.to(device)
                        target = target.to(device)
                        source = source.unsqueeze(0)
                        target = target.unsqueeze(0)
                        feature_map, pred = model(source)
                        pred = torch.atan(pred) * 2 / np.pi
                        #analyze_mouths(target, source, pred)
                        score = calc_perceptual_loss(target, pred, numpy=True)
                        if score <= thresh:
                            best_ixs.append(ix)
                    
                dataloader_train = reset_dataloader(best_ixs)
                dataloader = dataloader_train
                data_len = len(dataloader_train)
                model.train()
                saving_points = [int(data_len * x * save_at) -
                        1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1]
                print(saving_points)
            """
            
            """
            if phase == 'train' and epoch == 0:
                model.eval()
                pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
                data_csv = []
                
                for idx, (ixs, sources, targets) in pbar:
                    
                    #if idx == 100:
                    #    break

                    for ix, source, target in zip(ixs, sources, targets):
                        #print(sample)
                        #source, target = sample
                        source = source.to(device)
                        target = target.to(device)
                        source = source.unsqueeze(0)
                        target = target.unsqueeze(0)
                        score = calc_mouth_no_teeth(target)
                        data_csv.append([int(ix), score])
                        
                pd.DataFrame(data_csv, columns=['ix', 'score']).to_csv('scores1.csv')
                sys.exit()
            """
            
            
            if phase == 'train' and epoch == 0:
                thresh = -3.8
                data_csv = pd.read_csv('scores1.csv', index_col=0)
                best_seeds = data_csv[data_csv.score <= thresh].ix.to_list()
                #dataloader_train = reset_dataloader(best_seeds, is_seed=True)
                dataloader = dataloader_train
                data_len = len(dataloader_train)
                model.train()
                saving_points = [int(data_len * x * save_at) -
                        1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1]
                print(saving_points)
                if adjust_lr_every <= 10:
                    adjust_lr_every = adjust_lr_every * data_len * batch_size['train']
                adjust_lr_every = int(adjust_lr_every)
                

            pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
            for idx, (_, sources, targets) in pbar:
                #batch = batch.to(device)
                if phase == 'train':
                    sample_counter += batch_size['train']
                    adjust_lr(sample_counter, adjust_lr_every, batch_size['train'], optimizer)

                losses = []
                for source, target in zip(sources, targets):
                    #print(sample)
                    #source, target = sample
                    source = source.to(device)
                    target = target.to(device)
                    source = source.unsqueeze(0)
                    target = target.unsqueeze(0)

                    feature_map, pred = model(source)
                    pred = torch.atan(pred) * 2 / np.pi
                    vggface_loss = torch.cuda.FloatTensor([0]) if True else calc_vggface_loss(target, pred)
                    #vggface_loss = calc_perceptual_loss(target, pred)
                    l1_loss = calc_l1_loss(pred, target)
                    
                    #pred = normalize_after_reconet(pred)
                    #target = normalize_after_reconet(target)

                    #pred_features = Vgg16(pred)
                    #target_features = Vgg16(target)

                    content_loss = torch.cuda.FloatTensor([0]) if True else calc_content_loss(pred_features, target_features, alpha)

                    running_content_loss += content_loss.item()
                    running_vggface_loss += vggface_loss.item()
                    running_l1_loss += l1_loss.item()
                    #running_style_loss += style_loss.item()
                    #running_reg_loss += reg_loss.item()

                    img_loss = content_loss + vggface_loss + l1_loss # + style_loss + reg_loss
                    losses.append(img_loss)

                loss = sum(losses) / len(losses)
                scale_value = 1 / batch_size[phase] / max(idx, 1)
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss = (running_content_loss + running_vggface_loss + running_l1_loss) * scale_value

                pbar.set_description(
                    "Epoch: {}/{}, Phase: {}, Losses -> Content: {:.4f} VggFace: {:.4f} l1: {:.4f}".format(
                        epoch,
                        epochs,
                        phase,
                        running_content_loss * scale_value,
                        running_vggface_loss * scale_value,
                        running_l1_loss * scale_value
                    )
                )
                if checkpoint_path is not None and phase == 'train' and idx in saving_points:
                    #val_loss = (running_content_loss + running_vggface_loss) * scale_value
                    #if train_loss == val_loss:
                    #    val_loss = -1
                    torch.save(
                        model.state_dict(),
                        os.path.join(checkpoint_path, 'reconet_epoch_{}_train_loss_{:.4f}.pth'.format(
                            epoch,
                            train_loss,
                        )))
                    

                                    
                    
if __name__ == '__main__':
    # python3 train.py --data_path /home/konstantinlipkin/Anaconda_files/data_test --style_path /home/konstantinlipkin/Anaconda_files/data_path/some_class/image.jpg --phase first
    
    #python3 train.py --data_path ~/konst/data/moderation_resized --style_path ~/konst/ml-utils/training/neural_style_transfer/styles --checkpoint_path ~/konst/model_checkpoints/neural_style_transfer --phase first --batch_size 12 --manual_weights --alpha 5e5 --beta 1e3 --gamma 1e-6 --epochs 2 --save_at 0.5 --adjust_lr_every 0.3
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data",
                        help="Path to data root dir")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--model_path", default='',
                        help="Load existing model path")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument("--manual_weights", action='store_true',
                        help="Set manual weights for loss")
    parser.add_argument("--alpha", type=float, default=1e4,
                        help="Weight of content loss")
    parser.add_argument("--beta", type=float, default=1e5,
                        help="Weight of style loss")
    parser.add_argument("--gamma", type=float, default=1e-5,
                        help="Weight of style loss")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--frn", default=True, action='store_true',
                        help="Use Filter Response Normalization and TLU")
    parser.add_argument("--use_skip", action='store_true',
                        help="Use skip connections")
    parser.add_argument("--save_at", type=float, default=1,
                        help="Save checkpoint at current training stage, float in (0, 1)")
    parser.add_argument("--adjust_lr_every", type=float, default=1,
                        help="Lr decrease factor")
    parser.add_argument("--use_sim", action='store_true',
                        help="Use similarity weights for style pics")

    args = parser.parse_args()
    manual_weights = args.manual_weights

    if manual_weights:
        alpha = args.alpha
        beta = args.beta
        gamma = args.gamma
    else:
        alpha = 8e7  # previously 12, 2e10 // 1e4
        beta = 2.5e4  # 1e6 #11, // 1e5
        gamma = 3e-5  # previously -3 // 1e-5

    data_path = args.data_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    frn = args.frn
    use_skip = args.use_skip
    save_at = args.save_at
    adjust_lr_every = args.adjust_lr_every
    use_sim = args.use_sim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    IMG_SIZE = (600, 600) # 256, 256
    transform = T.Compose([
        T.Resize(IMG_SIZE, interpolation=Image.BICUBIC), # no resize if image were resized
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Lambda(lambda x: x.mul(2).sub(1))
    ])
    torch2PIL = T.ToPILImage()
    random.seed(0)
    train_size = 0.8
    data_len = len([f for f in os.listdir(data_path) if 'source' in f])
    mask_train = [random.random() < train_size for _ in range(data_len)]
    mask_test = [1 - m for m in mask_train]
    dataset_train = DatasetGenFace(
        data_path=data_path, 
        mask=mask_train,
        transform=additional_augmenation
    )
    dataset_test = DatasetGenFace(
        data_path=data_path, 
        mask=mask_test,
        transform=additional_augmenation
    )
    dataloader_train = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True
    )
    dataloader_test = DataLoader(
        dataset=dataset_test,
        batch_size=1,
        shuffle=True
    )
    model = ReCoNetMobile(a=1, b=1, frn=frn, use_skip=use_skip).to(device)
    print('Params, M: ', sum(p.numel() for p in model.parameters()) / 1e6)

    if model_path:
        #model.load_state_dict(torch.load(model_path, map_location=device))
        model = load_state_dict(model, model_path, device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    L2distance = nn.MSELoss().to(device)
    L2distancematrix = nn.MSELoss(reduction='none').to(device)
    Vgg16 = Vgg16().to(device)
    for param in Vgg16.parameters():
        param.requires_grad = False
        
    mtcnn = MTCNN(device=device)
    #vggface = InceptionResnetV1(pretrained='vggface2', device=device).eval()
    #face_detector = face_detection.build_detector(
    #    name='RetinaNetMobileNetV1',
    #    confidence_threshold=0.83,
    #    nms_iou_threshold=0.08,
    #    device=device
    #)
    #resnet = ResNet18().to(device)
    
    train(model, dataloader_train, dataloader_test, optimizer, L2distance, Vgg16, alpha, beta, gamma, epochs, checkpoint_path, device, save_at, adjust_lr_every)