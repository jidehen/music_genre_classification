import ast
import pandas
import os.path
import numpy as np
from track import Track
from load_audio import read_from_npy
import math
import tensorflow as tf

# Referencing - https://github.com/mdeff/fma/blob/master/utils.py
def get_data(track_path):

    genres = ['Experimental', 'Electronic', 'Rock', 'Instrumental', 'Pop', 'Folk', 'Hip-Hop', 'International',
              'Jazz', 'Classical', 'Country', 'Spoken', 'Blues', 'Soul-RnB', 'Old-Time / Historic', 'Easy Listening']
    genre_dict = {genre: id for id, genre in enumerate(genres)}
    tracks = pandas.read_csv(track_path, index_col=0, header=[0, 1])
    tracks = tracks['track'].to_numpy()
    id_to_features = read_from_npy()
    inputs = []
    labels = []
    for i in range(len(tracks)):
        track_id = i+2
        if track_id in id_to_features:
            track_genre = tracks[i][7]
            if track_genre in genre_dict and str(track_genre) != "nan":
                inputs.append(id_to_features[track_id].T) #transpose for lstm purposes
                labels.append(genre_dict[track_genre])

    labels = np.eye(len(genres))[labels]
    indices = tf.range(start=0, limit=len(inputs))
    shuffled = tf.random.shuffle(indices)
    inputs = tf.gather(inputs, shuffled)
    labels = tf.gather(labels, shuffled)

    split = int(len(inputs)*.85)

    train_inputs = inputs[:split]
    train_labels = labels[:split]
    test_inputs = inputs[split:]
    test_labels = labels[split:]
    return train_inputs, train_labels, test_inputs, test_labels

def get_batch(data, start, size):
	return data[start:start+size]