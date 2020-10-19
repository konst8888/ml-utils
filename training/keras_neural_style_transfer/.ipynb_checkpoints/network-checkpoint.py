from tensorflow.keras.applications.vgg16 import VGG16
#from keras.engine.topology import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from collections import namedtuple

class Vgg16(Model):
    def __init__(self, device='cpu'):
        super(Vgg16, self).__init__()
        vgg_pretrained = VGG16(
            weights='imagenet', 
            include_top=False)
        vgg_pretrained.load_weights('/home/konstantinlipkin/Anaconda_files/vgg_normalized_weights.h5')
        self.slice1 = Sequential()
        self.slice2 = Sequential()
        self.slice3 = Sequential()
        self.slice4 = Sequential()
        for x in range(3): # 3
            self.slice1.add(vgg_pretrained.layers[x]) # .to(device)
        for x in range(3, 6): # 3, 6
            self.slice2.add(vgg_pretrained.layers[x])
        for x in range(6, 10): # 6, 10
            self.slice3.add(vgg_pretrained.layers[x])
        for x in range(10, 14): # 10, 14
            self.slice4.add(vgg_pretrained.layers[x])
                
        for layer in self.layers:
            layer.trainable = False
            
    def call(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3)
        return out
    
"""
class Vgg16(Model):
    
    def __init__(self):
        
        vgg_pretrained = VGG16(weights='imagenet', include_top=False)
        self.vgg16_layer_dict = {layer.name:layer for layer in vgg_pretrained.layers}
        self.style_layers = ['block1_conv2','block2_conv2','block3_conv3','block4_conv3']
        
        self.slice1 = 
"""


class VGG19:
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    def __init__(self, data_path):
        data = scipy.io.loadmat(data_path)

        self.mean_pixel = np.array([123.68, 116.779, 103.939])

        self.weights = data['layers'][0]

    def preprocess(self, image):
        return image-self.mean_pixel

    def undo_preprocess(self,image):
        return image+self.mean_pixel

    def feed_forward(self, input_image, scope=None):
        net = {}
        current = input_image

        with tf.variable_scope(scope):
            for i, name in enumerate(self.layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels = self.weights[i][0][0][2][0][0]
                    bias = self.weights[i][0][0][2][0][1]

                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = bias.reshape(-1)

                    current = _conv_layer(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = _pool_layer(current)
                net[name] = current

        assert len(net) == len(self.layers)
        return net
