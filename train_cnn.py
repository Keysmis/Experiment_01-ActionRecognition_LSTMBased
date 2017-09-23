"""
Train on images split into directories. This assumes we've split
our videos into frames and moved them to their respective folders.

Based on:
https://keras.io/preprocessing/image/
and
https://keras.io/applications/
"""
from inception_v3_remix import InceptionV3
from keras.optimizers import SGD,Adam
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from data import DataSet

import keras.backend as K
from keras.models import Sequential, load_model
batch_size = 12
nb_classes = 101


img_rows, img_cols = 224, 224
img_channels = 3

img_dim = (img_channels, img_rows, img_cols) if K.image_dim_ordering() == "th" else (img_rows, img_cols, img_channels)

#dropout_rate = 0.0 # 0.0 for data augmentation



data = DataSet()
per_epoch=len(data.data) // batch_size

# Helper: Save the model.
checkpointer = ModelCheckpoint(
    filepath='./data/checkpoints/inception.{epoch:03d}-{val_loss:.2f}.hdf5',
    verbose=1,
    save_best_only=True)

# Helper: Stop when we stop learning.
early_stopper = EarlyStopping(patience=10)

# Helper: TensorBoard
tensorboard = TensorBoard(log_dir='./data/logs/')

def get_generators():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        horizontal_flip=True,
        rotation_range=10.,
        width_shift_range=0.2,
        height_shift_range=0.2)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        './data/train/',
        target_size=(299, 299),
        batch_size=64,
        classes=data.classes,
        class_mode='categorical')
    print train_generator.samples

    validation_generator = test_datagen.flow_from_directory(
        './data/test/',
        target_size=(299, 299),
        batch_size=64,
        classes=data.classes,
        class_mode='categorical')
    print validation_generator.samples

    return train_generator, validation_generator

def get_model():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    #x = Activation('relu',name='new_layer_activate')(x)
    x = Dense(101, activation='softmax',name='new_layer_dense')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=x)
    model.summary()
    return model

def get_top_layer_model(base_model):
    """Used to train just the top layers of the model."""
    # first: train only some the top layers (which were randomly initialized)
    for layer in base_model.layers[:312]:
        layer.trainable = False
    #print len(base_model.layers)

    # compile the model (should be done *after* setting layers to non-trainable)
    base_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    base_model.summary()

    return base_model

def get_mid_layer_model(model):
    """After we fine-tune the dense layers, train deeper."""
    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate
    model.compile(
        optimizer=SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'top_k_categorical_accuracy'])

    return model

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def main(weights_file):
    model = get_model()
    generators = get_generators()

    if weights_file is None:
        print("Loading network from ImageNet weights.")
        # Get and train the top layers.
        model = get_top_layer_model(model)
        model = train_model(model, 10, generators)
    else:
        print("Loading saved model: %s." % weights_file)
        model.load_weights(weights_file)

    # Get and train the mid layers.
    model = get_mid_layer_model(model)
    model = train_model(model, 1000, generators,
                        [checkpointer, early_stopper, tensorboard])


if __name__ == '__main__':
    weights_file = 'densenet121_weights_tf.h5'
    main(weights_file=None)
