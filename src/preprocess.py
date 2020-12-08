import ast
import pandas as pd
import os.path
import numpy as np
from track import Track
from load_audio import read_from_npy
import math
import tensorflow as tf
import collections

# Referencing - https://github.com/mdeff/fma/blob/master/utils.py


def get_data(track_path):

    all_genres = pd.read_csv("../data/fma_metadata/genres.csv", index_col=0)
    all_genres = all_genres.loc[all_genres['top_level'].unique()].sort_values('#tracks', ascending=False).index.values.tolist()[0:8]
    genre_dict = {genre: id for id, genre in enumerate(all_genres)}
    print("Top level genres: {}".format(all_genres))
    tracks = pd.read_csv(track_path, index_col=0, header=[0, 1])
    id_to_features = read_from_npy()
    inputs = []
    labels = []
    count = 0
    for id, row in tracks.iterrows():
        if id in id_to_features:
            track_genres = row['track']['genres_all'][1:-1]
            track_features = np.array(id_to_features[id])
            if len(track_genres) > 0:
                track_genres = [int(g.strip()) for g in track_genres.split(',')]
                genres_in_all = []
                for g in track_genres:
                    if g in all_genres:
                        inputs.append(track_features.T)
                        labels.append(genre_dict[g])
                        break #only add one top genre
    
    labels = np.eye(len(all_genres))[labels]
    
    indices = tf.range(start=0, limit=len(inputs))
    shuffled = tf.random.shuffle(indices)
    inputs = tf.gather(np.array(inputs), shuffled)
    labels = tf.gather(np.array(labels), shuffled)

    split = int(len(inputs)*.85)
    train_inputs = np.array(inputs[:split])
    train_labels = np.array(labels[:split])
    test_inputs = np.array(inputs[split:])
    test_labels = np.array(labels[split:])
    
    return train_inputs, train_labels, test_inputs, test_labels


def get_batch(data, start, size):
    return data[start:start+size]
