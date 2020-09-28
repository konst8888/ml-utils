"""
Chaanges by Kushagra :- commented 53 line,
Does the style tensor have batch size =2??? Or is that a mistake?
"""

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
import flowlib
from PIL import Image
import tqdm
import argparse

from model import ReCoNetMobile
from utilities import *
from network import *
from totaldata import *

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

def calc_content_loss(img_features, styled_features, alpha):
    out = L2distance(styled_features[2],
                                       img_features[2].expand(styled_features[2].shape))
    out *= alpha / (styled_features[2].shape[1] *
                 styled_features[2].shape[2] *
                 styled_features[2].shape[3])
                 
    return out
    
def calc_style_loss(style_GM, styled_features, STYLE_WEIGHTS, sim_weights, beta):
    out = 0
    for s_GM, sim_weight in zip(style_GM, sim_weights):
        current_loss = 0
        for i, weight in enumerate(STYLE_WEIGHTS):
            if i == 0:
                continue
            gram_s = s_GM[i]
            gram_img = gram_matrix(styled_features[i])
            #!!! below was gram_img1
            current_loss += float(weight) * L2distance(gram_img, gram_s.expand(
                gram_img.size()))
        out += current_loss * sim_weight
    out *= beta
    
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


def train_first_phase(model, dataloader, optimizer, L2distance, Vgg16, style_GM,
                      STYLE_WEIGHTS, alpha, beta, gamma, epochs, phase, checkpoint_path, device, save_at, adjust_lr_every):
    data_len = len(dataloader)
    batch_size = dataloader.batch_size
    sample_counter = 0
    if adjust_lr_every < 1:
    	adjust_lr_every = adjust_lr_every * data_len * batch_size
    adjust_lr_every = int(adjust_lr_every)
    for epoch in range(epochs):
        running_content_loss = 0
        running_style_loss = 0
        running_reg_loss = 0
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, (sample, _) in pbar:
            sample = sample.to(device)
            sample_counter += batch_size
            
            adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer)
             
            losses = []
            for img in sample:
                img = img.unsqueeze(0)
                feature_map, styled_img = model(img)
                styled_img = normalize_after_reconet(styled_img)
                img = normalize_after_reconet(img)

                styled_features = Vgg16(styled_img)
                img_features = Vgg16(img)

                content_loss = calc_content_loss(img_features, styled_features, alpha)
                content_loss += L2distance(styled_img, img)
                content_loss /= 2
                sim_weights = calc_sim_weights(img, style)
                style_loss = calc_style_loss(style_GM, styled_features, STYLE_WEIGHTS, sim_weights, beta)

                reg_loss = calc_reg_loss(styled_img, gamma)
                
                running_content_loss += content_loss.item()
                running_style_loss += style_loss.item()
                running_reg_loss += reg_loss.item()

                img_loss = content_loss + style_loss + reg_loss
                losses.append(img_loss)
                
            loss = sum(losses) / len(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scale_value = 1 / batch_size / max(idx, 1)
            pbar.set_description(
                "Epoch: {}/{} Losses -> Content: {:.4f} Style: {:.4f} Reg: {:.4f}".format(
                    epoch,
                    epochs,
                    running_content_loss * scale_value,
                    running_style_loss * scale_value,
                    running_reg_loss * scale_value
                )
            )
            if checkpoint_path is not None and idx in (int(data_len * save_at) - 1, data_len - 1):
                torch.save(
                    model.state_dict(),
                    os.path.join(checkpoint_path, 'reconet_phase_{}_epoch_{}_loss_{:.4f}.pth'.format(
                        phase,
                        epoch,
                        loss)))


def train_second_phase(model, dataloader, optimizer, L2distance, L2distancematrix, Vgg16, style_GM,
                       STYLE_WEIGHTS, alpha, beta, gamma, lambda_o, lambda_f, epochs, phase, checkpoint_path, device, save_at, adjust_lr_every):
    for epoch in range(epochs):
        running_content_loss = 0
        running_style_loss = 0
        running_reg_loss = 0
        running_ft_loss = 0
        running_ot_loss = 0
        pbar = tqdm.tqdm(enumerate(dataloader), total=len(dataloader))
        for idx, sample in pbar:
            img1, img2, mask, flow = [el.to(device) for el in sample]
            flow = -flow
            optimizer.zero_grad()
            if (idx + 1) % 1000 == 0:  # 500
                for param in optimizer.param_groups:
                    param['lr'] = max(param['lr'] / 1.2, 1e-4)

            feature_map1, styled_img1 = model(img1)
            feature_map2, styled_img2 = model(img2)
            styled_img1 = normalize_after_reconet(styled_img1)
            styled_img2 = normalize_after_reconet(styled_img2)
            img1, img2 = normalize_after_reconet(img1), normalize_after_reconet(img2)

            styled_features1 = Vgg16(styled_img1)
            styled_features2 = Vgg16(styled_img2)
            img_features1 = Vgg16(img1)
            img_features2 = Vgg16(img2)

            feature_flow = nn.functional.interpolate(
                flow, size=feature_map1.shape[2:], mode='bilinear')
            feature_flow[0, 0, :,
                         :] *= float(feature_map1.shape[2]) / flow.shape[2]
            feature_flow[0, 1, :,
                         :] *= float(feature_map1.shape[3]) / flow.shape[3]
            # print(flow.size(), feature_map1.shape[2:],feature_flow.size())
            feature_mask = nn.functional.interpolate(
                mask.view(1, 1, *IMG_SIZE), size=feature_map1.shape[2:], mode='bilinear')
            # print(feature_map1.size(), feature_flow.size())
            warped_fmap = warp(feature_map1, feature_flow, device)

            # #Changed by KJ to multiply with feature mask
            # # print(L2distancematrix(feature_map2, warped_fmap).size()) #Should be a matrix not number
            # # mean replaced sum
            f_temporal_loss = torch.sum(feature_mask *
                                        (L2distancematrix(feature_map2, warped_fmap)))
            f_temporal_loss *= lambda_f
            f_temporal_loss *= 1 / \
                (feature_map2.shape[1] * feature_map2.shape[2]
                 * feature_map2.shape[3])

            # # print(styled_img1.size(), flow.size())
            # # Removed unsqueeze methods in both styled_img1,flow in next line since already 4 dimensional
            warped_style = warp(styled_img1, flow, device)
            warped_image = warp(img1, flow, device)

            # print(img2.size())
            output_term = styled_img2[0] - warped_style[0]
            # print(output_term.shape, styled_img2.shape, warped_style.shape)
            input_term = img2[0] - warped_image[0]
            # print(input_term.size())
            # Changed the next few lines since dimension is 4 instead of 3 with batch
            # size=1
            input_term = 0.2126 * input_term[0, :, :] + 0.7152 * \
                input_term[1, :, :] + 0.0722 * input_term[2, :, :]
            input_term = input_term.expand(output_term.size())

            o_temporal_loss = torch.sum(
                mask * (L2distancematrix(output_term, input_term)))
            o_temporal_loss *= lambda_o
            o_temporal_loss *= 1 / (img1.shape[2] * img1.shape[3])

            content_loss = 0
            content_loss += L2distance(styled_features1[2],
                                       img_features1[2].expand(styled_features1[2].size()))
            content_loss += L2distance(styled_features2[2],
                                       img_features2[2].expand(styled_features2[2].size()))
            content_loss *= alpha / \
                (styled_features1[2].shape[1] *
                 styled_features1[2].shape[2] *
                 styled_features1[2].shape[3])

            style_loss = 0
            for i, weight in enumerate(STYLE_WEIGHTS):
                gram_s = style_GM[i]
                # print(styled_features1[i].size())
                gram_img1 = gram_matrix(styled_features1[i])
                gram_img2 = gram_matrix(styled_features2[i])
                # print(gram_img1.size(), gram_s.size())
                style_loss += float(weight) * (L2distance(gram_img1, gram_s.expand(
                    gram_img1.size())) + L2distance(gram_img2, gram_s.expand(gram_img2.size())))
            style_loss *= beta

            reg_loss = gamma * \
                (torch.sum(torch.abs(styled_img1[:, :, :, :-1] - styled_img1[:, :, :, 1:])) +
                 torch.sum(torch.abs(styled_img1[:, :, :-1, :] - styled_img1[:, :, 1:, :])))

            reg_loss += gamma * \
                (torch.sum(torch.abs(styled_img2[:, :, :, :-1] - styled_img2[:, :, :, 1:])) +
                 torch.sum(torch.abs(styled_img2[:, :, :-1, :] - styled_img2[:, :, 1:, :])))

            # print(f_temporal_loss.size(), o_temporal_loss.size(),
            # content_loss.size(), style_loss.size(), reg_loss.size())
            loss = f_temporal_loss + o_temporal_loss + content_loss + style_loss + reg_loss
            # loss = content_loss + style_loss
            loss.backward()
            optimizer.step()
            #
            #
            # if (idx+1)%1000 ==0 :
            # torch.save(model.state_dict(), '%s/final_reconet_epoch_%d_idx_%d.pth' %
            # ("runs/output", epoch, idx//1000))

            scale_value = 1 / batch_size / max(idx, 1)
            count = img2.shape[0]
            running_content_loss += content_loss.item() * count
            running_style_loss += style_loss.item() * count
            running_reg_loss += reg_loss.item() * count
            running_ft_loss += f_temporal_loss.item() * count
            running_ot_loss += o_temporal_loss.item() * count
            pbar.set_description(
                "Epoch: {}/{} Losses -> Content: {:.4f} Style: {:.4f} Reg: {:.4f} Feature: {:.4f} Output: {:.4f}".format(
                    epoch,
                    epochs,
                    running_content_loss * scale_value,
                    running_style_loss * scale_value,
                    running_reg_loss * scale_value,
                    running_ft_loss * scale_value,
                    running_ot_loss * scale_value
                )
            )
            # print('[%d/%d][%d/%d] SL: %.4f CL: %.4f FTL: %.4f OTL: %.4f RL: %.4f'
            #               % (epoch, epochs, idx, len(dataloader),
            # style_loss, content_loss , f_temporal_loss, o_temporal_loss, reg_loss))
        torch.save(
            model.state_dict(),
            os.path.join(checkpoint_path, 'reconet_phase_{}_epoch_{}_loss_{:.4f}.pth'.format(
                phase,
                epoch,
                loss)))


if __name__ == '__main__':
    # python3 train.py --data_path /home/konstantinlipkin/Anaconda_files/data_test --style_path /home/konstantinlipkin/Anaconda_files/data_path/some_class/image.jpg --phase first
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./data",
                        help="Path to data root dir")
    parser.add_argument("--style_path", help="Path folder containing style images")
    parser.add_argument("--checkpoint_path", type=str,
                        help="Checkpoints save path")
    parser.add_argument("--model_path", default='',
                        help="Load existing model path")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size")
    parser.add_argument(
        "--phase", type=str, help="Phase of training, required: {first, second} ")
    parser.add_argument("--manual_weights", action='store_true',
                        help="Set manual weights for loss")
    parser.add_argument("--alpha", type=float, default=1e4,
                        help="Weight of content loss")
    parser.add_argument("--beta", type=float, default=1e5,
                        help="Weight of style loss")
    parser.add_argument("--gamma", type=float, default=1e-5,
                        help="Weight of style loss")
    parser.add_argument("--lambda-f", type=float, default=1e5,
                        help="Weight of feature temporal loss")
    parser.add_argument("--lambda-o", type=float, default=2e5,
                        help="Weight of output temporal loss")
    parser.add_argument("--epochs", type=int, default=2,
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
        lambda_o = args.lambda_o
        lambda_f = args.lambda_f
    else:
        alpha = 1e13  # previously 12, 2e10 // 1e4
        beta = 1e10  # 1e6 #11, // 1e5
        gamma = 3e-2  # previously -3 // 1e-5
        lambda_o = 1e1  # // 2e5
        lambda_f = 1e-1  # // 1e5

    data_path = args.data_path
    style_path = args.style_path
    checkpoint_path = args.checkpoint_path
    model_path = args.model_path
    batch_size = args.batch_size
    phase = args.phase
    epochs = args.epochs
    lr = args.lr
    frn = args.frn
    use_skip = args.use_skip
    save_at = args.save_at
    adjust_lr_every = args.adjust_lr_every
    use_sim = args.use_sim
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # dataloader = DataLoader(FlyingChairsDataset("../FlyingChairs2/"),
    # batch_size=1)
    if phase == 'first':
        IMG_SIZE = (400, 400) # 256, 256
        transform = T.Compose([
#            T.Resize(IMG_SIZE), # no resize if image were resized
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x.mul(2).sub(1))
        ])
        dataset = COCODataset(data_path, transform)
        kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': True
        }
    elif phase == 'second':
        IMG_SIZE = (360, 640) #(768, 432)
        transform = T.Compose([
            T.Resize(IMG_SIZE),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x.mul(2).sub(1))
        ])
        mpi_path = os.path.join(data_path, 'mpi') + os.sep
        fc_path = os.path.join(data_path, 'FlyingChairs2') + os.sep
        dataset = ConsolidatedDataset(mpi_path, fc_path, transform)  # MPIDataset
        kwargs = {
            'dataset': dataset,
            'batch_size': 1,
            'shuffle': False
        }

    dataloader = DataLoader(**kwargs)
    model = ReCoNetMobile(frn=frn, use_skip=use_skip).to(device)
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    L2distance = nn.MSELoss().to(device)
    L2distancematrix = nn.MSELoss(reduction='none').to(device)
    Vgg16 = Vgg16().to(device)
    resnet = ResNet18().to(device)

    transform_style = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255)),
        normalize
    ])
    style = [Image.open(os.path.join(style_path, filename)) for filename in os.listdir(style_path) if not filename.endswith('checkpoints')]
    style = [transform_style(s) for s in style]
    # print(style.size())
    style = [s.unsqueeze(0).expand(
        1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device) for s in style]

    for param in Vgg16.parameters():
        param.requires_grad = False

    # [1e-1, 1e0, 1e1, 5e0, 1e1] not sure about what value to be deleted
    STYLE_WEIGHTS = [1e-1, 1e0, 1e1, 5e0]
    # STYLE_WEIGHTS = [1.0] * 4 in another implementation
    # print(style.size())
    styled_featuresR = [Vgg16(s) for s in style]
    # print(styled_featuresR[1].size())
    style_GM = [[gram_matrix(f) for f in styled_feature] 
    			for styled_feature in styled_featuresR]
    # print(len(style_GM))

    if phase == 'first':
        train_first_phase(model, dataloader, optimizer, L2distance, Vgg16, style_GM,
                          STYLE_WEIGHTS, alpha, beta, gamma, epochs, phase, checkpoint_path, device, save_at, adjust_lr_every)
    if phase == 'second':
        train_second_phase(model, dataloader, optimizer, L2distance, L2distancematrix, Vgg16, style_GM,
                           STYLE_WEIGHTS, alpha, beta, gamma, lambda_o, lambda_f, epochs, phase, checkpoint_path, device, save_at, adjust_lr_every)
