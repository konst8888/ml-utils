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
import os
import sys
import argparse

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, save_model

#import flowlib
from model import ReCoNetMobile
from utilities import *
from network import *
#from totaldata import *

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
    #out = (styled_features[2] - tf.reshape(img_features[2], styled_features[2].shape))
    out = tf.square(styled_features[2] - img_features[2])
    out = tf.reduce_sum(out)
    out *= alpha / (styled_features[2].shape[1] *
                 styled_features[2].shape[2] *
                 styled_features[2].shape[3])

    return out
    
def calc_style_loss(style_GM, styled_features, STYLE_WEIGHTS, sim_weights, beta):
    out = 0
    for s_GM, sim_weight in zip(style_GM, sim_weights):
        current_loss = 0
        for i, weight in enumerate(STYLE_WEIGHTS):
            #if i in (0, 1, np.nan):
            #    continue
            gram_s = s_GM[i]
            gram_img = gram_matrix(styled_features[i])
            #!!! below was gram_img1
            #current_loss += float(weight) * L2distance(gram_img, gram_s.expand(
            #    gram_img.size()))
            current_loss += float(weight) * tf.reduce_sum(tf.square(gram_img - gram_s))
        out += current_loss * sim_weight
    out *= beta

    out = tf.cast(out, tf.float32)
    return out
    
def calc_reg_loss(styled_img, gamma):
    out = gamma * \
        (tf.reduce_sum(tf.abs(styled_img[:, :, :-1, :] - styled_img[:, :, 1:, :])) +
         tf.reduce_sum(tf.abs(styled_img[:, -1:, :, :] - styled_img[:, 1:, :, :])))

    return out
    
def adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer):
    if (sample_counter + 1) % adjust_lr_every >= 0 \
    and (sample_counter + 1) % adjust_lr_every < batch_size:  # 500
        for param in optimizer.param_groups:
            param['lr'] = max(param['lr'] / 1.2, 1e-4)
            
class RunningLosses:
    
    def __init__(self):
        self.content_loss = 0.
        self.style_loss = 0.
        self.reg_loss = 0.
        self.counter = 0
    
    def update(self, losses_pbar):
        self.content_loss += float(losses_pbar['content'])
        self.style_loss += float(losses_pbar['style'])
        self.reg_loss += float(losses_pbar['reg'])
        self.counter += int(losses_pbar['count'])
    
    def get_losses(self):
        return list(
            map(lambda x: x / max(self.counter, 1), 
            [self.content_loss, self.style_loss, self.reg_loss])
        )

    def reset(self):
        self.__init__()
        
   
@tf.function
def compute_loss_and_grads(sample, rl):
    with tf.GradientTape() as tape:
        #losses = []
        #losses_pbar = tf.TensorArray(tf.float32, size=3)
        losses_pbar = {
            'content': 0.,
            'style': 0.,
            'reg': 0.,
            'count': 0
        }
        losses = tf.TensorArray(tf.float32, size=sample.shape[0])
        for i in range(sample.shape[0]):
            img = tf.expand_dims(sample[i], axis=0)
            feature_map, styled_img = model(img, training=True)
            styled_img = normalize_after_reconet(styled_img)
            img = normalize_after_reconet(img)

            tf.print(img, output_stream=sys.stderr)
            sys.exit()
            styled_features = Vgg16(styled_img)
            img_features = Vgg16(img)

            content_loss = calc_content_loss(img_features, styled_features, alpha)
            sim_weights = calc_sim_weights(img, style)
            style_loss = calc_style_loss(style_GM, styled_features, STYLE_WEIGHTS, sim_weights, beta)
            reg_loss = calc_reg_loss(styled_img, gamma)

            img_loss = content_loss + style_loss + reg_loss
            losses = losses.write(i, img_loss)
            
            losses_pbar['content'] += content_loss
            losses_pbar['style'] += style_loss
            losses_pbar['reg'] += reg_loss
            losses_pbar['count'] += 1
            
        loss = tf.reduce_sum(losses.stack()) #/ len(losses)    
        
    grads = tape.gradient(loss, model.trainable_weights)
    return loss, grads, losses_pbar


def train_first_phase(model, generator, optimizer, Vgg16, style_GM,
                      STYLE_WEIGHTS, alpha, beta, gamma, epochs, phase, checkpoint_path, save_at, adjust_lr_every):
    data_len = len(generator)
    batch_size = generator.batch_size
    sample_counter = 0
    if adjust_lr_every < 1:
        adjust_lr_every = adjust_lr_every * data_len * batch_size
    adjust_lr_every = int(adjust_lr_every)
    rl = RunningLosses()
    for epoch in range(epochs):
        running_content_loss = 0.
        running_style_loss = 0.
        running_reg_loss = 0.
        pbar = tqdm.tqdm(enumerate(generator), total=len(generator))
        #print(next(iter(pbar))[1].shape)
        for idx, sample in pbar:
            if idx > data_len:
                break
            sample_counter += batch_size
            #adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer)
            
            with tf.GradientTape() as tape:
                #losses = []
                #losses_pbar = tf.TensorArray(tf.float32, size=3)
                losses_pbar = {
                    'content': 0.,
                    'style': 0.,
                    'reg': 0.,
                    'count': 0
                }
                losses = tf.TensorArray(tf.float32, size=sample.shape[0])
                for i in range(sample.shape[0]):
                    img = tf.expand_dims(sample[i], axis=0)
                    img = img * 2 - 1
                    #img = np.array(img)
                    #print(img.max(), img.min())
                    feature_map, styled_img = model(img, training=True)
                    #styled_img = np.array(styled_img)
                    #print(styled_img.max(), styled_img.min())
                    styled_img = normalize_after_reconet(styled_img)
                    #styled_img = np.array(styled_img)
                    #print(styled_img.max(), styled_img.min())
                    img = normalize_after_reconet(img)
                    #img = np.array(img)
                    #print(img.max(), img.min())

                    #sys.exit()
                    styled_features = Vgg16(styled_img)
                    print(np.array(styled_features[0]).max(), np.array(styled_features[0]).min())
                    img_features = Vgg16(img)
                    print(np.array(img_features[0]).max(), np.array(img_features[0]).min())
                    sys.exit()

                    #for s in styled_features:
                    #    print(s.shape)
                    #sys.exit()
                    content_loss = calc_content_loss(img_features, styled_features, alpha)
                    sim_weights = calc_sim_weights(img, style)
                    style_loss = calc_style_loss(style_GM, styled_features, STYLE_WEIGHTS, sim_weights, beta)
                    reg_loss = calc_reg_loss(styled_img, gamma)

                    img_loss = content_loss + style_loss + reg_loss
                    losses = losses.write(i, img_loss)

                    losses_pbar['content'] += content_loss
                    losses_pbar['style'] += style_loss
                    losses_pbar['reg'] += reg_loss
                    losses_pbar['count'] += 1

                #loss = tf.reduce_sum(losses.stack()) / len(losses)    
                loss = K.mean(losses.stack())

            grads = tape.gradient(loss, model.trainable_weights)
            #loss, grads, losses_pbar = compute_loss_and_grads(sample, rl)

            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            rl.update(losses_pbar)
            losses = rl.get_losses()
            pbar.set_description(
                "Epoch: {}/{} Losses -> Content: {:.4f} Style: {:.4f} Reg: {:.4f}".format(
                    epoch,
                    epochs,
                    *losses
                )
            )

            if checkpoint_path is not None and idx in (int(data_len * save_at) - 1, data_len - 1):
                model.save_weights(
                    os.path.join(checkpoint_path, 'reconet_phase_{}_epoch_{}_loss_{:.4f}.h5'.format(
                        phase,
                        epoch,
                        loss)),
                    save_format='h5'
                )

if __name__ == '__main__':
    # python3 train.py --data_path /home/konstantinlipkin/Anaconda_files/data_test --style_path /home/konstantinlipkin/Anaconda_files/data_path/some_class/image.jpg --phase first
    
    #python3 train.py --data_path ~/konst/data/moderation_resized --style_path ~/konst/ml-utils/training/neural_style_transfer/styles --checkpoint_path ~/konst/model_checkpoints/neural_style_transfer --phase first --batch_size 12 --manual_weights --alpha 5e5 --beta 1e3 --gamma 1e-6 --epochs 2 --save_at 0.5 --adjust_lr_every 0.3
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
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    physical_devices = tf.config.list_physical_devices('GPU')
    for p in physical_devices:
        tf.config.experimental.set_memory_growth(p, True)

    # dataloader = DataLoader(FlyingChairsDataset("../FlyingChairs2/"),
    # batch_size=1)
    if phase == 'first':
        IMG_SIZE = (600, 600) # 256, 256
        transform = T.Compose([
            T.Resize(IMG_SIZE), # no resize if image were resized
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Lambda(lambda x: x.mul(2).sub(1))
        ])
        #dataset = COCODataset(data_path, transform)
        #kwargs = {
        #    'dataset': dataset,
        #    'batch_size': batch_size,
        #    'shuffle': True
        #}
        
    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)
    generator = datagen.flow_from_directory(
        data_path,
        target_size=IMG_SIZE,
        interpolation='bilinear',
        batch_size=batch_size,
        class_mode=None,
        classes=None
    )
    #for img in generator:
    #    print(img.shape)
    #dataloader = DataLoader(**kwargs)
    model = ReCoNetMobile(frn=frn, use_skip=use_skip) #.to(device)
    if model_path:
        model = load_model(model_path, compile=False)
        #model.load_state_dict(torch.load(model_path, map_location=device))

    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = keras.optimizers.Adam(lr=lr)

    #L2distance = nn.MSELoss().to(device)
    #L2distancematrix = nn.MSELoss(reduction='none').to(device)
    Vgg16 = Vgg16() #.to(device)
    #resnet = ResNet18().to(device)

    #transform_style = transforms.Compose([
    #    transforms.Resize(IMG_SIZE),
    #    transforms.ToTensor(),
    #    transforms.Lambda(lambda x: x.mul(255)),
    #    normalize
    #])
    style = [Image.open(os.path.join(style_path, filename)) for filename in os.listdir(style_path) if not filename.endswith('checkpoints')]
    #style = [transform_style(s) for s in style]
    # print(style.size())
    #style = [s.unsqueeze(0).expand(
    #    1, 3, IMG_SIZE[0], IMG_SIZE[1]).to(device) for s in style]
    style = [np.array(s.resize(IMG_SIZE)) for s in style]
    style = [np.expand_dims(normalize(s), axis=0) for s in style]
    
    # [1e-1, 1e0, 1e1, 5e0, 1e1] not sure about what value to be deleted
    STYLE_WEIGHTS = [1e-1, 1e0, 1e1, 5e0]
    # STYLE_WEIGHTS = [1.0] * 4 in another implementation
    styled_featuresR = [Vgg16(s) for s in style]
    for s in styled_featuresR[0]:
        print(tf.math.reduce_mean(s))
    sys.exit()

    # print(styled_featuresR[1].size())
    style_GM = [[gram_matrix(f) for f in styled_feature] 
    			for styled_feature in styled_featuresR]

    if phase == 'first':
        train_first_phase(model, generator, optimizer, Vgg16, style_GM,
                          STYLE_WEIGHTS, alpha, beta, gamma, epochs, phase, checkpoint_path, save_at, adjust_lr_every)
    if phase == 'second':
        pass