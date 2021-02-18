import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D, AveragePooling2D
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, constraints, initializers
from tensorflow_addons.layers import InstanceNormalization, SpectralNormalization


class ReflectionPad2d(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPad2d, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
    
    
def frn_layer_keras(x, beta, gamma, epsilon=1e-6):
    # x: Input tensor of shape [BxHxWxC].
    # tau, beta, gamma: Variables of shape [1, 1, 1, C].
    # eps: A scalar constant or learnable variable.
    # Compute the mean norm of activations per channel.
    nu2 = K.mean(K.square(x), axis=[1, 2], keepdims=True)
    # Perform FRN.
    x = x * 1 / K.sqrt(nu2 + K.abs(epsilon))
    # Return after applying the Offset-ReLU non-linearity.
    #return K.maximum(gamma * x + beta, tau)
    return gamma * x + beta

class FRN(Layer):
    def __init__(self,
                 epsilon=1e-6,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(FRN, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.gamma = None
        self.beta = None
        self.axis = -1
        
    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
        shape = (dim,)
        self.gamma = self.add_weight(shape=shape,
                                     name='gamma',
                                     initializer=self.gamma_initializer,
                                     regularizer=self.gamma_regularizer,
                                     constraint=self.gamma_constraint)
        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer,
                                    regularizer=self.beta_regularizer,
                                    constraint=self.beta_constraint)
        self.built = True

    def call(self, inputs, training=None):
        return frn_layer_keras(x=inputs, beta=self.beta, gamma=self.gamma, epsilon=self.epsilon)

    
class ConvLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride, activation=None, pad="same", transpose=False, use_bias=True):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if transpose:
            conv = tf.keras.layers.Conv2DTranspose
        else:
            conv = Conv2D
        if pad == "refl":
            layers = [
                ReflectionPad2d(kernel_size // 2),
                conv(filters=out_channels, kernel_size=kernel_size, strides=stride,
                     activation=activation, use_bias=use_bias)
                # SpectralNormalization(Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride,
                # activation=activation, padding="same"))
            ]
        else:
            layers = [
                conv(filters=out_channels, kernel_size=kernel_size, strides=stride,
                     activation=activation, padding=pad, use_bias=use_bias)
            ]
        self.layers = Sequential(layers)
    def call(self, x):
        return self.layers(x)
    
    
class ConvNormLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride, activation=True, frn=False, transpose=False, pad="refl"):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        if frn:
            layers = [
                ConvLayer(out_channels, kernel_size,
                          stride, transpose=transpose, pad=pad),
                FRN(),
            ]
            if activation:
                layers.append(TLU())
                # layers.append(ReLU())
        else:
            layers = [
                ConvLayer(out_channels, kernel_size,
                          stride, transpose=transpose, pad=pad),
                InstanceNormalization(axis=3, center=True, scale=True),
                # tf.keras.layers.BatchNormalization(axis=3, center=True, scale=True),
            ]
            if activation:
                layers.append(ReLU())
        self.layers = Sequential(layers)
        
    def call(self, x):
        return self.layers(x)
    
    
class ResLayer(Layer):
    def __init__(self, out_channels, kernel_size, reduction, frn=False):
        super().__init__()
        self.branch = Sequential([
            ReflectionPad2d(kernel_size // 2),
            Conv2D(filters=out_channels, kernel_size=kernel_size),
            InstanceNormalization(axis=3, center=True, scale=True) if not frn else FRN(), 
            tf.keras.layers.LeakyReLU(alpha=0.3),
            ReflectionPad2d(kernel_size // 2),
            Conv2D(filters=out_channels, kernel_size=kernel_size),
            InstanceNormalization(axis=3, center=True, scale=True) if not frn else FRN(),
            tf.keras.layers.LeakyReLU(alpha=0.3),
        ])
        self.ca_module = CALayer(out_channels, out_channels // reduction)
        # if frn:
        #     self.activation = TLU()
        # else:
        #     self.activation = ReLU()
    def call(self, x):
        y = self.branch(x)
        # y = self.activation(y)
        # return y
        x = x + self.ca_module(y)
        return x
class CALayer(tf.keras.layers.Layer):
    
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        # feature channel downscale and upscale --> channel weight
        self.conv_du = Sequential([
                Conv2D(channel // reduction, kernel_size=1, padding='valid', use_bias=True, activation='relu'),
                Conv2D(channel, kernel_size=1, padding='valid', use_bias=True, activation='sigmoid'),
        ])
        
    def call(self, x):
        y = self.avg_pool(x)
        y = tf.expand_dims(y, axis=1)
        y = tf.expand_dims(y, axis=1)
        #print(y.shape)
        y = self.conv_du(y)
        return x * y
class ConvNoTanhLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = Sequential([
            ConvLayer(out_channels, kernel_size, stride),
        ])
    def call(self, x):
        return self.layers(x)
class ConvTanhLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = Sequential([
            ConvLayer(out_channels, kernel_size, stride, activation="tanh"),
        ])
    def call(self, x):
        return self.layers(x)
    
class ReCoNetMobile(Model):
    def __init__(self, frn=True, a=0.5, b=0.75, use_skip=False):
        super().__init__()
        reduction = 16
        filter_counts = [int(a*x) for x in [32, 64, 128]]
        self.layers_encode = []
        for i in range(len(filter_counts)):
            self.layers_encode += [
                ConvLayer(filter_counts[i], 3, 2, pad="refl"),
                CALayer(filter_counts[i], reduction),
                # ConvLayer(filter_counts[i+1], 3, 2, pad="refl"),
                InstanceNormalization(axis=3, center=True, scale=True) if not frn else FRN(),
                tf.keras.layers.LeakyReLU(alpha=0.3),
            ]
        res_layer_count = int(b * 4)
        self.layers_middle = []
        for i in range(res_layer_count):
            self.layers_middle.append(
                ResLayer(filter_counts[-1], 3, reduction, frn=frn))
        self.layers_decode = []
        for i in range(len(filter_counts) - 1, -1, -1):
            self.layers_decode += [
                UpSampling2D(size=(2, 2), interpolation="bilinear"),
                ConvLayer(filter_counts[i], 3, 1, pad="refl"),
                InstanceNormalization(axis=3, center=True, scale=True) if not frn else FRN(),
                tf.keras.layers.LeakyReLU(alpha=0.3),
                # ConvNormLayer(filter_counts[i], 3, 1,
                #               frn=frn, transpose=False, pad="refl"),
                # ConvNormLayer(filter_counts[i] // 2, 3, 2, frn=frn, transpose=True, pad="same"),
                # tf.keras.layers.Conv2DTranspose(filter_counts[i], 3, 2, frn=frn),
                CALayer(filter_counts[i], reduction),
            ]
        self.conv = ConvNoTanhLayer(3, 3, 1)
        self.conv_skip = ConvLayer(3, 3, 1)
        # self.conv = ConvTanhLayer(3, 3, 1)
        # self.ca_module_resnets = CALayer(filter_counts[3], reduction)
        # self.ca_module_preprocess = CALayer(filter_counts[0], reduction)
        # self.ca_module_global = CALayer(filter_counts[0] * 2, reduction)
    def call(self, x, training=False):
        remember = [x]
        # remember = []
        # for layer in self.layers_preprocess:
        # x = layer(x)
        for layer in self.layers_encode:
            # x = layer(x)
            if isinstance(layer, ConvLayer):
                if layer.stride == 2:
                    remember.append(x)
            x = layer(x)
        # y = x
        for layer in self.layers_middle:
            x = layer(x)
        # x += self.ca_module_resnets(y)
        for ind, layer in enumerate(self.layers_decode):
            # if isinstance(layer, ConvNormLayer) and layer.stride == 2:
            #     x = tf.concat([x, remember.pop(-1)], -1)
            # x = layer(x)
                # y = x
            if ind > 0 and isinstance(self.layers_decode[ind - 1], UpSampling2D):
                x = tf.concat([x, remember.pop(-1)], -1)
            # if isinstance(layer, ConvNormLayer):
            #     y = x
            x = layer(x)
            # if isinstance(layer, CALayer):
            #     x = y + x
        # x = self.ca_module_global(x)
        return self.conv(x) + self.conv_skip(remember.pop(-1))