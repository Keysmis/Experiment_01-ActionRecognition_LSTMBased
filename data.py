"""
Class for managing our data.
"""
import csv
import numpy as np
import random
import glob
import os.path
import pandas as pd
import sys
import operator
from processor import process_image
from keras.utils import np_utils

class DataSet():

    def __init__(self, seq_length=40, class_limit=None, image_shape=(299, 299, 3)):
        """Constructor.
        seq_length = (int) the number of frames to consider
        class_limit = (int) number of classes to limit the data to.
            None = no limit.
        """
        self.seq_length = seq_length
        self.class_limit = class_limit
        self.sequence_path = '/home/wpc/sequences/'
        self.max_frames = 400  # max number of frames a video can have for us to use it

        # Get the data.
        self.data = self.get_data()

        # Get the classes.
        self.classes = self.get_classes()

        # Now do some minor data cleaning.
        self.data = self.clean_data()

        self.image_shape = image_shape

    @staticmethod
    def get_data():
        """Load our data from file."""
        with open('./data/data_file.csv', 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)

        return data

    def clean_data(self):
        """Limit samples to greater than the sequence length and fewer
        than N frames. Also limit it to classes we want to use."""
        data_clean = []
        for item in self.data:
            if int(item[3]) >= 16 and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])

        # Sort them.
        classes = sorted(classes)
        #print classes

        # Return.
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = np_utils.to_categorical(label_encoded, len(self.classes))
        label_hot = label_hot[0]  # just get a single row

        return label_hot

    def split_train_test(self):
        """Split the data into train and test groups."""
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, batch_Size, train_test, data_type, concat=False):
        """
        This is a mirror of our generator, but attempts to load everything into
        memory so we can train way faster.
        """
        # Get the right dataset.
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Getting %s data with %d samples." % (train_test, len(data)))

        X, y = [], []
        for row in data:
            #print row
            for i in range(3):

                sequence = self.get_extracted_sequence(data_type, row, i)
                #print (sequence.shape)

                if sequence is None:
                    print("Can't find sequence. Did you generate them?")
                    raise

                if concat:
                    # We want to pass the sequence back as a single array. This
                    # is used to pass into a CNN or MLP, rather than an RNN.
                    sequence = np.concatenate(sequence).ravel()

                X.append(sequence)
                y.append(self.get_class_one_hot(row[1]))

        return np.array(X), np.array(y)

    def test_frame(self, batch_size):
        """Return a generator that we can use to test on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = test
        #random.shuffle(data)
        #print data

        print("Creating test generator with %d samples." % len(data))
        #t=0

        while 1:
            X, y = [], []
            X_image_path = []

            # reset_t_and_shuffle_data= int(len(data)*0.8)
            # if t >= reset_t_and_shuffle_data:
            #     t = 0
            #     random.shuffle(data)
            # #i = 0

            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None
                #t = t + 1
                #print t

                # Get a random sample.
                sample = random.choice(data)
                # sample = data[t]
                # t =t + 1
                #print sample
                #data.remove(sample)
                #print data
                #print np.array(data).shape

                #get 2 sets of frame flow from sample
                for i in range(2):
                    #t = t + 1
                    #print t
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames)
                    #frames_group = self.get_frames_data(frames)
                    frames_group = self.get_frames_of_sixteen(frames)
                    X_image_path.append(frames_group)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames_group)
                    X.append(sequence)
                    y.append(self.get_class_one_hot(sample[1]))

            yield np.array(X), np.array(y)

    def frame_generator(self, batch_size):
        """Return a generator that we can use to train on. There are
        a couple different things we can return:

        data_type: 'features', 'images'
        """
        # Get the right dataset for the generator.
        train, test = self.split_train_test()
        data = train
        random.shuffle(data)
        #print data

        print("Creating train generator with %d samples." %  len(data))
        t = 0

        while 1:
            #t = 0
            #print len(data)
            reset_t_and_shuffle_data= int(len(data) * 0.8)
            if t >= reset_t_and_shuffle_data:
                t = 0
                random.shuffle(data)


            X, y = [], []
            X_image_path = []


            # Generate batch_size samples.
            for _ in range(batch_size):
                # Reset to be safe.
                sequence = None
                #sample = random.choice(data)
                sample = data[t]
                t = t + 1
                for i in range(2):
                    # Get and resample frames.
                    frames = self.get_frames_for_sample(sample)
                    frames = self.rescale_list(frames)
                    # frames_group = self.get_frames_data(frames)
                    frames_group = self.get_frames_of_sixteen(frames)
                    X_image_path.append(frames_group)

                    # Build the image sequence
                    sequence = self.build_image_sequence(frames_group)
                    X.append(sequence)
                    y.append(self.get_class_one_hot(sample[1]))


            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def get_extracted_sequence(self, data_type, sample, i):
        """Get the saved extracted features."""
        filename = sample[2]
        # path = self.sequence_path + filename + '-' + str(self.seq_length) + \
        #     '-' + data_type + '.txt'
        # path = self.sequence_path + filename + '-16-' + str(self.seq_length) +'-'+ str(i) +'-' + data_type + '.txt'
        path = self.sequence_path + filename + '-16-40-' + str(i) + '-' + data_type + '.txt'
        print ('path:%s' % path)
        if os.path.isfile(path):
            # Use a dataframe/read_csv for speed increase over numpy.
            features = pd.read_csv(path, sep=" ", header=None)
            return features.values
        else:
            return None

    @staticmethod
    def get_frames_for_sample(sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = './data/' + sample[0] + '/' + sample[1] + '/'
        filename = sample[2]
        images = sorted(glob.glob(path + filename + '*jpg'))
        return images

    @staticmethod
    def get_filename_from_image(filename):
        parts = filename.split('/')
        return parts[-1].replace('.jpg', '')

    @staticmethod
    def get_frames_of_sixteen(input_list, num_frames_per_clip=16):
        ''' Given a directory containing extracted frames, return a video clip of
        (num_frames_per_clip) consecutive frames as a list of np arrays

        Args
          num_frames_per_clip: sequence_length of the video clip

        Returns
          video: numpy, video clip with shape
            [sequence_length, width, height, channels]
        '''

        if (len(input_list) < num_frames_per_clip):
            return None

        s_index = random.randint(1, len(input_list)-num_frames_per_clip)

        output_list = input_list[s_index:s_index+num_frames_per_clip]
        output_sort = sorted(output_list)

        return output_sort

    @staticmethod
    def rescale_list(input_list, size=40):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        #assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size
        if skip ==0:
            skip = skip + 1
        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]

