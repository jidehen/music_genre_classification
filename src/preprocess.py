import ast
import pandas as pd
import os.path
import numpy as np
from track import Track
from load_audio import read_from_npy
import math
import tensorflow as tf
import collections
import string

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
            track_features = np.array(id_to_features[id])[:50].T
            track_featurse = track_features[0:100]
            if len(track_genres) > 0:
                track_genres = [int(g.strip()) for g in track_genres.split(',')]
                genres_in_all = []
                track_listens = row['track']['listens']
                album_listens = row['album']['listens']
                favorites = row['track']['favorites']
                interests = row['track']['interest']
                title = row['track']['title']
                artist = row['artist']['name']
                for g in track_genres:
                    if g in all_genres:
                        inputs.append(Track(id, title, artist, track_features, track_listens, album_listens, favorites, interests))
                        labels.append(genre_dict[g])
                        break #only add one top genre

    labels = np.eye(len(all_genres))[labels]

    split = int(len(inputs)*.85)
    train_inputs = np.array(inputs[:split])
    train_labels = np.array(labels[:split])
    test_inputs = np.array(inputs[split:])
    test_labels = np.array(labels[split:])

    return train_inputs, train_labels, test_inputs, test_labels


def get_batch(data, start, size):
    return data[start:start+size]

def make_numerical_lists(train_inputs, test_inputs):
    train_track_listens = [x.track_listens for x in train_inputs]
    # train_album_listens = [x.album_listens for x in train_inputs]
    train_favorites = [x.favorites for x in train_inputs]
    train_interests = [x.interests for x in train_inputs]
    train_inputs = np.stack((train_track_listens, train_favorites, train_interests), axis=1)

    test_track_listens = [x.track_listens for x in test_inputs]
    # test_album_listens = [x.album_listens for x in test_inputs]
    test_favorites = [x.favorites for x in test_inputs]
    test_interests = [x.interests for x in test_inputs]
    test_inputs = np.stack((test_track_listens, test_favorites, test_interests), axis=1)

    return train_inputs, test_inputs

def make_feature_lists(train_inputs, test_inputs):
    return [x.features for x in train_inputs], [y.features for y in test_inputs]

def make_char_dict(inputs):
    titles = [x.title.replace(" ", "").lower() for x in inputs] # get a list of titles
    titles = np.asarray(titles).flatten()

    all_chars = []
    for word in titles:
        for char in word:
            if char.isalpha():
                all_chars.append(char)

    vocab_set = sorted(set(all_chars))
    vocab_dict = {val:id for id, val in enumerate(vocab_set)}

    title_sequences = []
    for word in titles:
        for char in word:
            if char.isalpha():
                title_sequences.append(vocab_dict[char])

    return title_sequences, vocab_dict #a list of titles in the form of int sequences
