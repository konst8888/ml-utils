import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd

import os
import sys

from tqdm import tqdm
import random

from dataset import DataGenerator
from models.generator import Generator
from models.discriminator import Discriminator
from utils import RunningLosses

def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  image = tf.cast(image, tf.float32)

  return image

def resize(input_image, real_image, height, width):
  #input_image = tf.image.resize(input_image, [height, width],
  #                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  #real_image = tf.image.resize(real_image, [height, width],
  #                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, 512, 512, 3])

  return cropped_image[0], cropped_image[1]

# normalizing the images to [-1, 1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to 286 x 286 x 3
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  # randomly cropping to 256 x 256 x 3
  #input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(input_file, real_file):
  input_image = load(input_file)
  real_image = load(real_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(input_file, real_file):
  input_image = load(input_file)
  real_image = load(real_file)
  #input_image, real_image = resize(input_image, real_image,
  #                                 IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

loss_object = tf.keras.losses.BinaryCrossentropy(
        from_logits=True, 
        label_smoothing=0.1
)

def set_seed(seed=0):
  os.environ['PYTHONHASHSEED']=str(seed)
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # new flag present in tf 2.0+
  random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)


def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
  gen_l1_loss = LAMBDA * l1_loss
  total_gen_loss = gan_loss + gen_l1_loss

  return total_gen_loss, gan_loss, gen_l1_loss


def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss
  #print(total_disc_loss)
  #sys.exit()

  return total_disc_loss

def reset_dataloader(train_ds, ixs, is_seed=False):
    train_ds.filter_index(ixs, is_seed)

    return train_ds

def adjust_lr(sample_counter, adjust_lr_every, batch_size, optimizer):
    if (sample_counter + 1) % adjust_lr_every >= 0 \
    and (sample_counter + 1) % adjust_lr_every < batch_size:  # 500
        new_lr = max(optimizer.learning_rate / 1.2, 2e-4)
        K.set_value(optimizer.learning_rate, new_lr)

@tf.function
def train_step(
        generator, 
        discriminator, 
        generator_optimizer, 
        discriminator_optimizer, 
        input_image, target, epoch, phase, batch_size
        ):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    is_train = phase == 'train'
    losses_pbar = dict()

    gen_output = generator(input_image, training=is_train)
    #sys.exit()
    disc_real_output = discriminator([input_image, target], training=is_train)
    disc_generated_output = discriminator([input_image, gen_output], training=is_train)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    if is_train:
      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))

  losses_pbar['gen_total'] = gen_total_loss
  losses_pbar['gen_gan'] = gen_gan_loss
  losses_pbar['gen_l1'] = gen_l1_loss
  losses_pbar['disc'] = disc_loss
  losses_pbar['count'] = batch_size

  return losses_pbar
  
def fit(
    cfg, 
    train_ds, 
    test_ds, 
    generator, 
    discriminator, 
    generator_optimizer, 
    discriminator_optimizer):
    
  checkpoint_path = cfg.checkpoint_path
  adjust_lr_every = cfg.adjust_lr_every
  save_at = cfg.save_at
  batch_size = cfg.batch_size
  epochs = cfg.epochs
  start_epoch = cfg.start_epoch

  rl = RunningLosses('gen_total', 'gen_gan', 'gen_l1', 'disc')
  data_len = len(train_ds)
  sample_counter = 0
  if adjust_lr_every <= 10:
      adjust_lr_every = adjust_lr_every * data_len * batch_size
  adjust_lr_every = int(adjust_lr_every)
  saving_points = [int(data_len * x * save_at) -
                 1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1]
  print(saving_points)
  for epoch in range(epochs):
    
    for phase in ['train', 'valid']:
      ds = train_ds if phase == 'train' else test_ds
        
      if phase == 'train' and epoch == 0:
        thresh = -3.8
        data_csv = pd.read_csv('scores1.csv', index_col=0)
        best_seeds = data_csv[data_csv.score <= thresh].ix.to_list()
        train_ds = reset_dataloader(train_ds, best_seeds, is_seed=True)
        ds = train_ds
        data_len = len(train_ds)
        saving_points = [int(data_len * x * save_at) -
               1 for x in range(1, int(1 / max(save_at, 0.01)))] + [data_len - 1]
        print(saving_points)
        if adjust_lr_every <= 10:
            adjust_lr_every = adjust_lr_every * data_len * batch_size['train']
        adjust_lr_every = int(adjust_lr_every)
      
      pbar = tqdm(enumerate(ds), total=len(ds))
      for idx, (input_image, target) in pbar:
        #if random.random() < 0.05:
        #    from PIL import Image
        #    Image.fromarray(((input_image[0].numpy() + 1) / 2 * 255).astype('uint8')).save(f'i_{idx}.jpg')
        #    Image.fromarray(((target[0].numpy() + 1) / 2 * 255).astype('uint8')).save(f't_{idx}.jpg')
        sample_counter += batch_size
        adjust_lr(sample_counter, adjust_lr_every, batch_size, generator_optimizer)
        adjust_lr(sample_counter, adjust_lr_every, batch_size, discriminator_optimizer)

        losses_pbar = train_step(
          generator, 
          discriminator, 
          generator_optimizer, 
          discriminator_optimizer,
           input_image, target, epoch, phase, batch_size
           )
        rl.update(losses_pbar)
        losses = rl.get_losses()
        #if idx % 100 == 0:
        #    print(input_image.numpy().min(), input_image.numpy().max())
        #    pic = generator(tf.expand_dims(input_image[0], axis=0), training=False)
        #    print(pic.numpy().min(), pic.numpy().max())
        pbar.set_description(
            "Epoch: {}/{} Phase: {} Losses -> gen_total: {:.4f} gen_gan: {:.4f} gen_l1: {:.4f} disc: {:.4f}".format(
                epoch + start_epoch,
                epochs + start_epoch,
                phase,
                *losses
            )
        )
        if phase == 'train' and checkpoint_path is not None and idx in saving_points:
          discriminator.save_weights(
            os.path.join(checkpoint_path, 'disc_epoch_{}_loss_{}.h5'.format(
                  epoch + start_epoch,
                  round(float(losses[3]), 4),
              )), save_format='h5')
          generator.save_weights(
            os.path.join(checkpoint_path, 'gen_epoch_{}_loss_{}.h5'.format(
                  epoch + start_epoch,
                  round(float(losses[0]), 4),
              )), save_format='h5')

IMG_WIDTH = 600
IMG_HEIGHT = 600
SIZE = (IMG_WIDTH, IMG_HEIGHT)
LAMBDA = 1 # 40

def train(cfg): # drop resize

  data_path = cfg.data_path
  batch_size = cfg.batch_size
  lr = cfg.lr
  checkpoint_path = cfg.checkpoint_path

  set_seed(seed=0)
  physical_devices = tf.config.list_physical_devices('GPU')
  print(physical_devices)
  for p in physical_devices:
      tf.config.experimental.set_memory_growth(p, True)

  data_len = len([f for f in os.listdir(data_path) if 'source_' in f])
  mask_train = [random.random() < cfg.train_size for _ in range(data_len)]
  #mask_train = [0] * (data_len // 2) + [1] * (data_len // 2)
  mask_test = [1 - m for m in mask_train]
  train_dataset = DataGenerator(data_path, mask_train, batch_size, load_image_train, SIZE)
  test_dataset = DataGenerator(data_path, mask_test, batch_size, load_image_test, SIZE)

  generator = Generator(a=1.5, b=1.5, frn=cfg.frn, use_skip=cfg.use_skip) # a=1.5, b=1.5
  generator(np.ones((1, *SIZE, 3)), training=False)
  discriminator = Discriminator(SIZE)
  discriminator([np.ones((1, *SIZE, 3)), np.ones((1, *SIZE, 3))], training=False)
    
  if cfg.gen_path and cfg.disc_path:
    try:
        generator.load_weights(os.path.join(checkpoint_path, cfg.gen_path))
    except ValueError:
        print('Load weights partly...')
        generator.load_weights(os.path.join(checkpoint_path, cfg.gen_path), by_name=True, skip_mismatch=True)
    discriminator.load_weights(os.path.join(checkpoint_path, cfg.disc_path))

  #EPOCHS = 150
  print(generator.count_params() / 1e6)
  #sys.exit()

  generator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5) # 2e-4
  discriminator_optimizer = tf.keras.optimizers.Adam(lr, beta_1=0.5)

  fit(cfg, train_dataset, test_dataset, generator, discriminator, 
    generator_optimizer, discriminator_optimizer)
