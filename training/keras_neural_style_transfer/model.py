import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.models import Sequential, Model
#from keras.engine.topology import Layer
#from keras.engine import InputSpec
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Conv2D, ReLU, UpSampling2D
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers, constraints, initializers
from tensorflow_addons.layers import InstanceNormalization #, TLU

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


class ConvLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride):
        super().__init__()

        self.layers = Sequential([
            ReflectionPad2d(kernel_size // 2),
            Conv2D(filters=out_channels, kernel_size=kernel_size, strides=stride)
        ])

    def call(self, x):
        return self.layers(x)
    

def frn_layer_keras(x, tau, beta, gamma, epsilon=1e-6):
    # x: Input tensor of shape [BxHxWxC].
    # tau, beta, gamma: Variables of shape [1, 1, 1, C].
    # eps: A scalar constant or learnable variable.
    # Compute the mean norm of activations per channel.
    nu2 = K.mean(K.square(x), axis=[1, 2], keepdims=True)
    # Perform FRN.
    x = x * 1 / K.sqrt(nu2 + K.abs(epsilon))
    # Return after applying the Offset-ReLU non-linearity.
    return K.maximum(gamma * x + beta, tau)


class FRN(Layer):

    def __init__(self,
                 epsilon=1e-6,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 tau_initializers='zeros',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 tau_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 tau_constraint=None,
                 **kwargs):
        super(FRN, self).__init__(**kwargs)
        self.supports_masking = True
        self.epsilon = epsilon
        self.beta_initializer = initializers.get(beta_initializer)
        self.tau_initializer = initializers.get(tau_initializers)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.tau_regularizer = regularizers.get(tau_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        self.tau_constraint = constraints.get(tau_constraint)
        self.tau = None
        self.gamma = None
        self.beta = None
        self.axis = -1

    def build(self, input_shape):
        dim = input_shape[self.axis]
        self.input_spec = InputSpec(ndim=len(input_shape), axes={self.axis: dim})
        shape = (dim,)
        self.tau = self.add_weight(shape=shape,
                                   name='tau',
                                   initializer=self.tau_initializer,
                                   regularizer=self.tau_regularizer,
                                   constraint=self.tau_constraint)
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
        return frn_layer_keras(x=inputs, tau=self.tau, beta=self.beta, gamma=self.gamma, epsilon=self.epsilon)
    

class TLU(tf.keras.layers.Layer):
    r"""Thresholded Linear Unit.

    An activation function which is similar to ReLU
    but with a learned threshold that benefits models using FRN(Filter Response
    Normalization). Original paper: https://arxiv.org/pdf/1911.09737.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Arguments:
        affine: `bool`. Whether to make it TLU-Affine or not
            which has the form $\max(x, \alpha*x + \tau)$`
    """

    def __init__(
        self,
        affine: bool = False,
        tau_initializer = "zeros",
        tau_regularizer = None,
        tau_constraint = None,
        alpha_initializer = "zeros",
        alpha_regularizer = None,
        alpha_constraint = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.affine = affine
        self.tau_initializer = tf.keras.initializers.get(tau_initializer)
        self.tau_regularizer = tf.keras.regularizers.get(tau_regularizer)
        self.tau_constraint = tf.keras.constraints.get(tau_constraint)
        if self.affine:
            self.alpha_initializer = tf.keras.initializers.get(alpha_initializer)
            self.alpha_regularizer = tf.keras.regularizers.get(alpha_regularizer)
            self.alpha_constraint = tf.keras.constraints.get(alpha_constraint)

    def build(self, input_shape):
        #param_shape = list(input_shape[1:])
        #print(param_shape)
        param_shape = list(input_shape[3:])
        self.tau = self.add_weight(
            shape=param_shape,
            name="tau",
            initializer=self.tau_initializer,
            regularizer=self.tau_regularizer,
            constraint=self.tau_constraint,
            synchronization=tf.VariableSynchronization.AUTO,
            aggregation=tf.VariableAggregation.MEAN,
        )
        #self.tau = np.array([1, 2, 3], dtype=np.float32)
        if self.affine:
            self.alpha = self.add_weight(
                shape=param_shape,
                name="alpha",
                initializer=self.alpha_initializer,
                regularizer=self.alpha_regularizer,
                constraint=self.alpha_constraint,
                synchronization=tf.VariableSynchronization.AUTO,
                aggregation=tf.VariableAggregation.MEAN,
            )

        #axes = {i: input_shape[i] for i in range(1, len(input_shape))}
        axes = {3: input_shape[3]}
        self.input_spec = tf.keras.layers.InputSpec(ndim=len(input_shape), axes=axes)
        self.built = True

    def call(self, inputs):
        v = self.alpha * inputs if self.affine else 0
        #shape = inputs.shape[1:]
        #tau = tf.ones(shape) * self.tau
        return tf.maximum(inputs, self.tau + v)

    def get_config(self):
        config = {
            "tau_initializer": tf.keras.initializers.serialize(self.tau_initializer),
            "tau_regularizer": tf.keras.regularizers.serialize(self.tau_regularizer),
            "tau_constraint": tf.keras.constraints.serialize(self.tau_constraint),
            "affine": self.affine,
        }

        if self.affine:
            config["alpha_initializer"] = tf.keras.initializers.serialize(
                self.alpha_initializer
            )
            config["alpha_regularizer"] = tf.keras.regularizers.serialize(
                self.alpha_regularizer
            )
            config["alpha_constraint"] = tf.keras.constraints.serialize(
                self.alpha_constraint
            )

        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape
    

class ConvNormLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride, activation=True, frn=False):
        super().__init__()

        if frn:
            layers = [
                ConvLayer(out_channels, kernel_size, stride),
                FRN(),
            ]
            if activation:
                layers.append(TLU())
        else:
            layers = [
                ConvLayer(out_channels, kernel_size, stride),
                InstanceNormalization(axis=3, center=True, scale=True),
            ]
            if activation:
                layers.append(ReLU())

        self.layers = Sequential(layers)

    def call(self, x):
        return self.layers(x)
    

class ResLayer(Layer):

    def __init__(self, out_channels, kernel_size, frn=False):
        super().__init__()
        self.branch = Sequential([
            ConvNormLayer(out_channels, kernel_size, stride=1, frn=frn),
            ConvNormLayer(out_channels, kernel_size, stride=1, activation=False, frn=frn)
        ])

        if frn:
            self.activation = TLU()
        else:
            self.activation = ReLU()

    def call(self, x):
        x = x + self.branch(x)
        x = self.activation(x)
        return x
    
    
class ConvNoTanhLayer(Layer):
    def __init__(self, out_channels, kernel_size, stride):
        super().__init__()
        self.layers = Sequential([
            ConvLayer(out_channels, kernel_size, stride),
        ])

    def call(self, x):
        return self.layers(x)
    
class Encoder(Layer):
    def __init__(self, a, b, frn=False, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        filter_counts = list(map(lambda x: int(a * x), [
            32, 48, 64
        ]))
        if not use_skip:
            self.layers = Sequential([
                ConvNormLayer(filter_counts[0], 3, 1, frn=frn),
                ConvNormLayer(filter_counts[1], 3, 2, frn=frn),
                ConvNormLayer(filter_counts[2], 3, 2, frn=frn),
            ])
            res_layer_count = int(b * 4)
            [self.layers.add(ResLayer(filter_counts[2], 3, frn=frn)) 
            for i in range(res_layer_count)]
        else:
            self.layers_first = Sequential([
                ConvNormLayer(filter_counts[0], 3, 1, frn=frn),
            ])
            self.layers_second = Sequential([
                ConvNormLayer(filter_counts[1], 3, 2, frn=frn),
                ConvNormLayer(filter_counts[2], 3, 2, frn=frn),
            ])
            res_layer_count = int(b * 4)
            [self.layers_second.add(ResLayer(filter_counts[2], 3, frn=frn)) 
            for i in range(res_layer_count)]


    def call(self, x):
        if not self.use_skip:
            return self.layers(x)
        else:
            x = self.layers_first(x)
            f_map = x
            x = self.layers_second(x)
            return x, f_map

class Decoder(Layer):
    def __init__(self, a, b, frn=False, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        filter_counts = list(map(lambda x: int(a * x), [
            64, 48, 32
        ]))
        if not use_skip:
            self.layers = Sequential([
                UpSampling2D(size=(2, 2), interpolation="nearest"),
                ConvNormLayer(filter_counts[1], 3, 1, frn=frn),
                UpSampling2D(size=(2, 2), interpolation="nearest"),
                ConvNormLayer(filter_counts[2], 3, 1, frn=frn),
                ConvNoTanhLayer(3, 3, 1)
            ])
        else:
            self.layers_first = Sequential([
                UpSampling2D(size=(2, 2), interpolation="nearest"),
                ConvNormLayer(filter_counts[1], 3, 1, frn=frn),
                UpSampling2D(size=(2, 2), interpolation="nearest"),
                ConvNormLayer(filter_counts[2], 3, 1, frn=frn),
            ])
            self.layers_second = Sequential([
                ConvNoTanhLayer(3, 3, 1)
            ])
            self.conv = ConvLayer(filter_counts[2], 3, 1)

    def call(self, x):
        if not self.use_skip:
            return self.layers(x)
        else:
            x, f_map = x
            x = self.layers_first(x)
            f_map = self.conv(f_map)
            #x = torch.cat([x, f_map], dim=1)
            x += f_map
            x = self.layers_second(x)
            return x

        
class ReCoNetMobile(Model):
    def __init__(self, frn=True, a=0.5, b=0.75, use_skip=False):
        super().__init__()
        self.use_skip = use_skip
        self.encoder = Encoder(a=a, b=b, frn=frn, use_skip=use_skip)
        self.decoder = Decoder(a=a, b=b, frn=frn, use_skip=use_skip)

    def call(self, x):
        if not self.use_skip:
            x = self.encoder(x)
            features = x
            x = self.decoder(x)
        else:
            x, f_map = self.encoder(x)
            features = x
            x = self.decoder((x, f_map))
        return (features, x)
