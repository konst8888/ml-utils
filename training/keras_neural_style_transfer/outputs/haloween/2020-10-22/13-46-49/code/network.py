from tensorflow.keras.applications.vgg16 import VGG16
#from keras.engine.topology import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from collections import namedtuple

class Vgg16(Model):
    def __init__(self, device='cpu', vgg_path=None):
        super(Vgg16, self).__init__()
        vgg_pretrained = VGG16(weights='imagenet', include_top=False)
        if vgg_path is not None:
            vgg_pretrained.load_weights(vgg_path)
        self.slice1 = Sequential()
        self.slice2 = Sequential()
        self.slice3 = Sequential()
        self.slice4 = Sequential()
        for x in range(3):
            self.slice1.add(vgg_pretrained.layers[x]) # .to(device)
        for x in range(3, 6):
            self.slice2.add(vgg_pretrained.layers[x])
        for x in range(6, 10):
            self.slice3.add(vgg_pretrained.layers[x])
        for x in range(10, 14):
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
    
