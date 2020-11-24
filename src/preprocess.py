import ast
import pandas
import os.path
import numpy as np
from track import Track

# Referencing - https://github.com/mdeff/fma/blob/master/utils.py
def get_data(track_path, feature_path):

    inputs = []
    labels = []

    tracks = pandas.read_csv(track_path, index_col=0, header=[0, 1])
    tracks = tracks['track'].to_numpy()

    features = pandas.read_csv(feature_path, index_col=0, header=[0, 1, 2])
    features = features.to_numpy()

    for i in range(len(tracks) - 1):
        track_id = i+2
        title = tracks[i][19]
        genres_all = tracks[i][9]
        feature = features[i]
        inputs.append(Track(track_id, title, genres_all, feature))
        labels.append(genres_all)

    return inputs, labels
