
import os
os.environ['KERAS_BACKEND']= 'tensorflow'
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from models import ResearchModels
from data import DataSet
import time



def train(data_type, seq_length, model, saved_model_extractnet=None,saved_model_lstm=None,
          concat=False, class_limit=None, image_shape=None,
          load_to_memory=False):
    # Set variables.
    nb_epoch = 1000
    batch_size = 32

    # Helper: Save the model.
    checkpointer = ModelCheckpoint(
        filepath='./data/checkpoints/' + model + '16-40-conv-lstm-mixed-' + data_type + \
            '.{epoch:03d}-{val_loss:.3f}.hdf5',
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    tb = TensorBoard(log_dir='./data/logs')

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=10)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger('./data/logs/' + model + '-' + 'training-' + \
        str(timestamp) + '.log')

    # Get the data and process it.
    if image_shape is None:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit
        )
    else:
        data = DataSet(
            seq_length=seq_length,
            class_limit=class_limit,
            image_shape=image_shape
        )

    # Get samples per epoch.
    # Multiply by 0.7 to attempt to guess how much of data.data is the train set.
    steps_per_epoch = (len(data.data) * 0.7) // batch_size

    if load_to_memory:
        # Get data.
        X, y = data.get_all_sequences_in_memory(batch_size, 'train', data_type)
        print X.shape
        X_test, y_test = data.get_all_sequences_in_memory(batch_size, 'test', data_type)
    else:
        # Get generators.
        generator = data.frame_generator(batch_size, 'train', data_type, concat)
        val_generator = data.test_frame(batch_size, 'test', data_type, concat)

    # Get the model.
    rm = ResearchModels(len(data.classes), model, seq_length, saved_model_extractnet=saved_model_extractnet,
                         saved_model_lstm=saved_model_lstm)

    # Fit!
    if load_to_memory:
        # Use standard fit.
        rm.model.fit(X, y,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            shuffle=False,
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            epochs=nb_epoch)
    else:
        # Use fit generator.
        rm.model.fit_generator(
            generator=generator,
            steps_per_epoch=steps_per_epoch,
            epochs=nb_epoch,
            verbose=1,
            callbacks=[checkpointer, tb, early_stopper, csv_logger],
            validation_data=val_generator,
            validation_steps=20)


def main():
    """These are the main training settings. Set each before running
    this file."""
    model = 'lstm'  # see `models.py` for more
    saved_model_extractnet = 'inception.002-1.31.hdf5' # None or weights file
    saved_model_lstm = 'crnn16-40-phlstm-images.013-2.468.hdf5' # lstm model weights
    class_limit = 50  # int, can be 1-101 or None
    seq_length = 16
    load_to_memory = True  # pre-load the sequences into memory

    # Chose images or features and image shape based on network.
    if model == 'conv_3d' or model == 'crnn':
        data_type = 'images'
        image_shape = (299, 299, 3)
        load_to_memory = False
    else:
        data_type = 'features'
        image_shape = None


    train(data_type, seq_length, model, saved_model_extractnet=saved_model_extractnet,saved_model_lstm=saved_model_lstm,
          class_limit=class_limit, image_shape=image_shape,
          load_to_memory=load_to_memory)

if __name__ == '__main__':

    main()
