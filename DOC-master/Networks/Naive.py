from __future__ import absolute_import, division, print_function, unicode_literals
from Networks.NNInterface import NNInterface
from tensorflow.python.keras.applications import VGG16


import tensorflow as tf

IMG_SIZE = 160 # All images will be resized to 160x160

class Naive(NNInterface):
    def __init__(self):
        super(Naive, self).__init__()
        self.network = VGG16(weights='imagenet')




        print("Naive network constructed")
        self.network.summary()



    def call(self, x):

        return self.network(x)