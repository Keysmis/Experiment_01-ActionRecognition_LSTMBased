from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.models import Model, load_model
from keras.layers import Input
import numpy as np
from inception_v3_remix import InceptionV3
from densenet121 import DenseNet
class Extractor():
    def __init__(self, weights=None):
        """Either load pretrained from imagenet, or load our saved
        weights from our own training."""

        self.weights = weights  # so we can check elsewhere which model

            # Load the model first.
        self.model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
        #self.model = DenseNet(reduction=0.5, classes=1000, weights_path=weights)

        # Then remove the top so we get features not predictions.
        # From: https://github.com/fchollet/keras/issues/2371
        self.model.layers.pop()
        self.model.layers.pop()  # two pops to get to pool layer
        self.model.outputs = [self.model.layers[-1].output]
        self.model.output_layers = [self.model.layers[-1]]
        self.model.layers[-1].outbound_nodes = []

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Get the prediction.
        features = self.model.predict(x)

        if self.weights is None:
            # For imagenet/default network:
            features = features[0]
        else:
            # For loaded network:
            features = features[0]

        return features
