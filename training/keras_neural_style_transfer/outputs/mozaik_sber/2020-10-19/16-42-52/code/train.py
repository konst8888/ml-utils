import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import os
import sys
import argparse
import random

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model, save_model

from model import ReCoNetMobile
from utilities import gram_matrix, normalize, normalize_after_reconet, IMG_SIZE
from network import Vgg16



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
        
def set_seed(seed=8):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
    tf.set_random_seed(seed)

def train(cfg):
    def train_first_phase(alpha, beta, gamma, epochs, checkpoint_path, save_at, adjust_lr_every):
        epochs, end_at = int(epochs // 1), epochs % 1
        if end_at > 0:
            epochs += 1

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
            out = tf.reduce_mean(out)
            out *= alpha

            return out


        def calc_style_loss(style_GM, styled_features, STYLE_WEIGHTS, sim_weights, beta):
            out = 0
            for s_GM, sim_weight in zip(style_GM, sim_weights):
                current_loss = 0
                for i, weight in enumerate(STYLE_WEIGHTS):

                    gram_s = s_GM[i]
                    gram_img = gram_matrix(styled_features[i])

                    current_loss += float(weight) * \
                        tf.reduce_mean(tf.square(gram_img - gram_s))
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
                    param['lr'] = max(param['lr'] / 5., 1e-4)

        @tf.function
        def compute_loss_and_grads(model, sample, rl):
            with tf.GradientTape() as tape:
                losses_pbar = {
                    'content': 0.,
                    'style': 0.,
                    'reg': 0.,
                    'count': 0
                }
                losses = tf.TensorArray(tf.float32, size=sample.shape[0])
                for i in range(sample.shape[0]):
                    img = sample[i]
                    img = img * 2 - 1
                    img = tf.expand_dims(img, axis=0)

                    feature_map, styled_img = model(img, training=True)
                    styled_img = normalize_after_reconet(styled_img)
                    img = normalize_after_reconet(img)
                    styled_features = vgg16(styled_img)
                    img_features = vgg16(img)
                    content_loss = calc_content_loss(
                        img_features, styled_features, alpha)
                    sim_weights = calc_sim_weights(img, style)
                    style_loss = calc_style_loss(
                        style_GM, styled_features, STYLE_WEIGHTS, sim_weights, beta)
                    reg_loss = calc_reg_loss(styled_img, gamma)

                    img_loss = content_loss + style_loss + reg_loss
                    losses = losses.write(i, img_loss)

                    losses_pbar['content'] += content_loss
                    losses_pbar['style'] += style_loss
                    losses_pbar['reg'] += reg_loss
                    losses_pbar['count'] += 1

                loss = tf.reduce_sum(losses.stack())  # / len(losses)

            grads = tape.gradient(loss, model.trainable_weights)
            return loss, grads, losses_pbar

        data_len = len(generator)
        batch_size = generator.batch_size
        sample_counter = 0
        saving_points = [int(data_len * x * save_at) -
                         1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1, 0]

        print(saving_points)
        if adjust_lr_every < 1:
            adjust_lr_every = adjust_lr_every * data_len * batch_size

        adjust_lr_every = int(adjust_lr_every)
        rl = RunningLosses()
        for epoch in range(epochs):
            running_content_loss = 0.
            running_style_loss = 0.
            running_reg_loss = 0.
            pbar = tqdm.tqdm(enumerate(generator), total=len(generator))
            # print(next(iter(pbar))[1].shape)
            for idx, sample in pbar:
                if idx > data_len:
                    break
                sample_counter += batch_size
                #adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer)

                loss, grads, losses_pbar = compute_loss_and_grads(model, sample, rl)

                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                rl.update(losses_pbar)
                losses = rl.get_losses()
                logs = [epoch, epochs, losses[0], losses[1], losses[2]]
                pbar.set_description(
                    "Epoch: {}/{} Losses -> Content: {:.4f} Style: {:.4f} Reg: {:.4f}".format(
                        *logs
                    )
                )

                if checkpoint_path is not None and idx in saving_points:
                    model.save_weights(
                        os.path.join(checkpoint_path, 'epoch_{}_{:.2f}_loss_{:.4f}_c_{:.4f}_s_{:.4f}_r_{:.4f}.h5'.format(
                            epoch,
                            idx / data_len,
                            loss,
                            losses[0],
                            losses[1],
                            losses[2],)),
                        save_format='h5'
                    )
                if idx == int(data_len * end_at) - 1 and epoch == epochs - 1:
                    return

    os.makedirs(cfg.model.checkpoint_path, exist_ok=True)

    set_seed(seed=cfg.model.seed)
    physical_devices = tf.config.list_physical_devices('GPU')
    for p in physical_devices:
        tf.config.experimental.set_memory_growth(p, True)

    datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

    generator = datagen.flow_from_directory(
        cfg.dataset.path,
        target_size=IMG_SIZE,
        interpolation='bilinear',
        batch_size=cfg.training.batch_size,
        class_mode=None,
        classes=None
    )

    model = ReCoNetMobile(frn=cfg.model.frn, use_skip=cfg.model.use_skip)
    model(np.ones((1, *IMG_SIZE, 3)), training=False)
    if cfg.model.path:
        model.load_weights(cfg.model.path)
        
    #for layer in model.submodules:
    #    if isinstance(layer, tf.keras.layers.Conv2D):
    #        layer.kernel_initializer = 'ones'
    #        layer.bias_initializer = 'ones'
    #    if isinstance(layer, FRN):
    #        layer.gamma_initializer = 'ones'
    #        layer.beta_initializer = 'ones'
    #    if isinstance(layer, TLU):
    #        layer.beta_initializer = 'ones'


    optimizer = keras.optimizers.Adam(lr=cfg.training.lr)


    vgg16 = Vgg16(vgg_path = cfg.model.vgg_path)  # .to(device)

    style = [Image.open(os.path.join(cfg.dataset.style_path, filename)) for filename in os.listdir(
        cfg.dataset.style_path) if not filename.endswith('checkpoints')]
    style = [np.array(s.resize(IMG_SIZE)) for s in style]
    style = [np.expand_dims(normalize(s), axis=0) for s in style]

    use_sim = cfg.training.use_sim
    STYLE_WEIGHTS = cfg.model.style_weights
    styled_featuresR = [vgg16(s) for s in style]
    for s in styled_featuresR[0]:
        print(tf.math.reduce_mean(s))

    style_GM = [[gram_matrix(f) for f in styled_feature]
                for styled_feature in styled_featuresR]

    train_first_phase(cfg.model.loss_weights.alpha,
                      cfg.model.loss_weights.beta, cfg.model.loss_weights.gamma,
                      cfg.training.epochs, cfg.model.checkpoint_path,
                      cfg.training.save_at, cfg.training.adjust_lr_every)
