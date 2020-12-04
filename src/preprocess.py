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

    tracks_data = pandas.read_csv(track_path, index_col=0, header=[0, 1])
    tracks = tracks_data['track'].to_numpy()
    # album = tracks_data['album'].to_numpy()

    genres = ['Experimental', 'Electronic', 'Rock', 'Instrumental', 'Pop', 'Folk', 'Hip-Hop', 'International',
              'Jazz', 'Classical', 'Country', 'Spoken', 'Blues', 'Soul-RnB', 'Old-Time / Historic', 'Easy Listening']
    genre_dict = {genre: id for id, genre in enumerate(genres)}


    # id_to_features = read_from_npy()

    inputs = []
    labels = []

    for i in range(len(tracks)):
        id = i+2
        # if id in id_to_features:
        genre_label = tracks[i][7]
        if genre_label in genre_dict and str(genre_label) != "nan":
            genre = genre_dict[genre_label]
            # feats = id_to_features[id].T #transpose for lstm purposes
            feats = None
            listens = tracks[i][14]
            duration = tracks[i][5]
            comments = tracks[i][1]
            favorites = tracks[i][6]
            interest = tracks[i][11]

            inputs.append(Track(id, genre, feats, listens, duration, comments, favorites, interest))
            labels.append(genre)

    labels = np.eye(len(genres))[labels]

    split = int(len(inputs)*.85)

    train_inputs = inputs[:split]
    train_labels = labels[:split]
    test_inputs = inputs[split:]
    test_labels = labels[split:]

    return train_inputs, train_labels, test_inputs, test_labels

def get_batch(data, start, size):
	return data[start:start+size]
