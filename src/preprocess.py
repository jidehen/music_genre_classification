import ast
import pandas
import os.path
import numpy as np
from track import Track
from load_audio import read_from_npy
import math
import tensorflow as tf
import collections

# Referencing - https://github.com/mdeff/fma/blob/master/utils.py


def get_data(track_path):

    genres = ['Experimental', 'Electronic', 'Rock', 'Pop', 'Folk', 'Hip-Hop', 'International', 'Classical']
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
            if track_genre in genre_dict:
                # transpose for lstm purposes
                inputs.append(np.array(id_to_features[track_id]).T)
                labels.append(genre_dict[track_genre])

    #Downsample
    # genre_counter = collections.Counter(labels) #Count the frequency of each genre
    # least_frequent_genre = genre_counter.most_common()[-1] #Find the lowest-freq genre
    # max = least_frequent_genre[1] #Establish count of lowest genre as maximum for all genre sizes
    # genre_counts = [0 for g in genres]
    # tmp_inputs = []
    # tmp_labels = []
    
    # for i in range(len(labels)):
    #     if genre_counts[labels[i]] < max:
    #         tmp_labels.append(labels[i])
    #         tmp_inputs.append(inputs[i])
    #         genre_counts[labels[i]] += 1
    
    # labels = tmp_labels
    # inputs = tmp_inputs

    # genre_counter = collections.Counter(labels)
    # print("list length : {}".format(len(labels)))
    # print("least frequent: {}".format(least_frequent_genre))
    # print("Frequency of the elements in the List : ", genre_counter)

    # print("Eyeing labels...")

    labels = np.eye(len(genres))[labels]
    # indices = tf.range(start=0, limit=len(inputs))
    # shuffled = tf.random.shuffle(indices)
    # inputs = tf.gather(inputs, shuffled)
    # labels = tf.gather(labels, shuffled)

    print(len(inputs))
    split = int(len(inputs)*.85)
    train_inputs = np.array(inputs[:split])
    train_labels = np.array(labels[:split])
    test_inputs = np.array(inputs[split:])
    test_labels = np.array(labels[split:])
    
    return train_inputs, train_labels, test_inputs, test_labels


def get_batch(data, start, size):
    return data[start:start+size]
