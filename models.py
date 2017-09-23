"""
A collection of models we'll use to attempt to classify videos.
"""
from keras.layers import Dense, Flatten, Dropout
from keras import regularizers
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D,
    MaxPooling2D)
from collections import deque
import sys
from inception_v3_remix import InceptionV3

from keras.metrics import top_k_categorical_accuracy

from PhasedLSTM import PhasedLSTM

def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)
def acc_top1(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=1)

class ResearchModels():
    def __init__(self, nb_classes, model, seq_length,
                 saved_model_extractnet=None,saved_model_lstm=None, features_length=2048):
        """
        `model` = one of:
            lstm
            crnn
            mlp
            conv_3d
        `nb_classes` = the number of classes to predict
        `seq_length` = the length of our video sequences
        `saved_model` = the path to a saved Keras model to load
        """

        # Set defaults.
        self.seq_length = seq_length
        # self.load_model = load_model
        self.saved_model_densenet = saved_model_extractnet
        self.saved_model_lstm = saved_model_lstm
        self.nb_classes = nb_classes
        self.feature_queue = deque()

        # Set the metrics. Only use top k if there's a need.
        metrics = ['accuracy']

        if self.nb_classes >= 10:
            metrics.append('top_k_categorical_accuracy')
            metrics.append(acc_top3)
            metrics.append(acc_top1)

        # Get the appropriate model.

        if model == 'lstm':
            print("Loading LSTM model.")
            self.input_shape = (seq_length, features_length)
            self.model = self.lstm()
        elif model == 'crnn':
            print("Loading CRNN model.")
            self.input_shape = (seq_length, 299, 299, 3)
            self.model = self.crnn()
        # elif model == 'conv_3d':
        #     print("Loading Conv3D")
        #     self.input_shape = (seq_length, 224, 224, 3)
        #     self.model = self.conv_3d()
        else:
            print("Unknown network.")
            sys.exit()

        # Now compile the network.
        optimizer = Adam(lr=1E-5)
        #optimizer = RMSprop(lr=0.001,decay=0.1)  # aggressively small learning rate
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                           metrics=metrics)

    def lstm(self):
        """Build a simple LSTM network. We pass the extracted features from
        our CNN to this model predomenently."""
        # Model.
        print self.input_shape
        model = Sequential()
        model.add(PhasedLSTM(2048, return_sequences=True, input_shape=self.input_shape,dropout=0.5))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.summary()

        return model


    def crnn(self):
        """Build a CNN into RNN.
        Starting version from:
        https://github.com/udacity/self-driving-car/blob/master/
            steering-models/community-models/chauffeur/models.py
        """
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
        # this is the model we will train

        base_model.layers.pop()
        base_model.layers.pop()
        #dense_model.layers.pop()
        base_model.output_layers = [base_model.layers[-1]]
        base_model.outputs = [base_model.layers[-1].output]
        base_model.layers[-1].outbound_nodes = []
        for layer in base_model.layers:
            layer.trainable = False
        base_model.summary()
        #print dense_model.output_layers
        #print dense_model.output_shape
        model = Sequential()
        model.add(TimeDistributed(base_model,input_shape=self.input_shape))
        model.add(PhasedLSTM(256, return_sequences=True))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Dropout(0.5))
        model.add(Dense(self.nb_classes, activation='softmax'))
        model.summary()
        model.load_weights(self.saved_model_lstm,by_name=True)

        return model
    #
    # def conv_3d(self):
    #     """
    #     Build a 3D convolutional network, based loosely on C3D.
    #         https://arxiv.org/pdf/1412.0767.pdf
    #     """
    #     # Model.
    #     model = Sequential()
    #     model.add(Conv3D(
    #         32, (7,7,7), activation='relu', input_shape=self.input_shape
    #     ))
    #     model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    #     model.add(Conv3D(64, (3,3,3), activation='relu'))
    #     model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    #     model.add(Conv3D(128, (2,2,2), activation='relu'))
    #     model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2)))
    #     model.add(Flatten())
    #     model.add(Dense(256))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(256))
    #     model.add(Dropout(0.2))
    #     model.add(Dense(self.nb_classes, activation='softmax'))
    #
    #     return model
