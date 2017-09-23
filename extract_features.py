"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from. This is useful is you don't want to wait to
extract all 101 classes. For instance, set class_limit = 8 to just
extract features for the first 8 (alphabetical) classes in the dataset.
Then set the same number when training models.
"""
import numpy as np
import os.path
from data import DataSet
from extractor import Extractor
from tqdm import tqdm

# Set defaults.
seq_length = 40
class_limit = 50  # Number of classes to extract. Can be 1-101 or None for all.

# Get the dataset.
data = DataSet(seq_length=seq_length, class_limit=class_limit)

# get the model.
model = Extractor()

# Loop through data.
pbar = tqdm(total=len(data.data)*3)
for video in data.data:


    frames = data.get_frames_for_sample(video)

    # Now downsample to just the ones we need.
    frames = data.rescale_list(frames, seq_length)
    # print frames
    # print len(frames)
    for i in range(3):
        frames_of_sixteen = data.get_frames_of_sixteen(frames)

        # Get the path to the sequence for this video.
        path = '/home/wpc/sequences/' + video[2] + '-16-' + str(seq_length) +'-'+ str(i) +'-features.txt'

        # Check if we already have it.
        if os.path.isfile(path):
            pbar.update(1)
            continue


    # Now loop through and extract features to build the sequence.
        sequence = []
        for image in frames_of_sixteen:
            features = model.extract(image)
            sequence.append(features)

    # Save the sequence.
        np.savetxt(path, sequence)

        pbar.update(1)

pbar.close()
