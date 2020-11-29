import ast
import pandas
import os.path
import numpy as np
from track import Track

# Referencing - https://github.com/mdeff/fma/blob/master/utils.py
def get_data(track_path, feature_path):

    inputs = []
    labels = []

    genres = ['Soul-RnB', 'Old-Time / Historic', 'Rock', 'Instrumental', 'Blues', 'Country', 'Electronic', 'Folk', 'Pop', 'Hip-Hop', 'Spoken', 'International', 'Classical', 'Jazz']
    genre_dict = {genre:id for id, genre in enumerate(genres)}

    tracks = pandas.read_csv(track_path, index_col=0, header=[0, 1])
    tracks = tracks['track'].to_numpy()

    features = pandas.read_csv(feature_path, index_col=0, header=[0, 1, 2])
    features = features.to_numpy()

    for i in range(1):
        if tracks[i][7] in genre_dict:
            # track_id = i+2
            # title = tracks[i][19]
            # genre = genre_dict[tracks[i][7]]
            # feature = features[i]
            inputs.append(features[i])
            labels.append(genre_dict[tracks[i][7]])

    rand = np.random.permutation(len(inputs))
    inputs = inputs[rand]
    labels = labels[rand]

    train_inputs = inputs[:int((len(data)*.85))]
    train_labels = labels[:int((len(data)*.85))]
    test_inputs = inputs[int((len(data)*.85)):]
    test_labels = labels[int((len(data)*.85)):]

    for i in range(len(train_inputs)):
        print("inputs val: ", train_inputs[i])
        print("labels val: ", train_labels[i])

    return train_inputs, train_labels, test_inputs, test_labels
