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
    titles = []
    max_len = 0
    for track in inputs:
        title = track.title
        curr_max = len(title)
        max_len = max(max_len, curr_max)
        titles.append(title.lower())

    title_lists = []
    idx = 0
    for title in titles:
        padding = ["*"] * (max_len - len(title))
        title_lists.append(list(title) + padding)



    # Set window size
    WINDOW_SIZE = max_len
    #
    # # Build Vocabulary (char id's)
    chars = [j for i in title_lists for j in i]
    vocab = set(chars) # collects all unique words in our dataset (vocab)
    char2id = {w: i for i, w in enumerate(list(vocab))} # maps each word in our vocab to a unique index (label encode)


    # s = map(lambda x: x.split(), title_lists)

    #Create Skipgram Data
    data = []
    for title in title_lists:
        title_ids = []
        for char in title:
            curr_char = char2id[char]
            title_ids.append(curr_char)
        data.append(title_ids)

    # return title_sequences, vocab_dict #a list of titles in the form of int sequences
    return char2id, data
