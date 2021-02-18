import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def Discriminator(size):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[*size, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[*size, 3], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)


class DiscModel(tf.keras.Model):

  def __call__(self, x, training):
    inp, tar = x
    x = tf.concat([inp, tar], axis=-1)
    return super().__call__(x, training)


def Discriminator1(size):
  inp = tf.keras.layers.Input(shape=[*size, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[*size, 3], name='target_image')

  input_layer = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)
  base_model = MobileNetV2(include_top=False,
                                  pooling='avg', input_shape=(*size, 3))
  dense_input = tf.keras.layers.Input(shape=(*size, 6))
  dense_filter = tf.keras.layers.Conv2D(3, 3, padding='same')(dense_input)
  output = base_model(dense_filter)
  output = tf.keras.layers.Dense(
      2,  activation=None)(output)

  model = DiscModel(inputs=dense_input, outputs=output) # tf.keras.Model

  return model
