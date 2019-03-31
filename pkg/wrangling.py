'''
data wrangling
'''

from random import sample as rand_sample
from copy import deepcopy
import csv

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd # pylint: disable=import-error
from maps import FrozenMap  # pylint: disable=import-error
from tqdm import tqdm_notebook  # pylint: disable=import-error

from .constants import FLOAT_FEATURE_LIST, SCALABLE_VALUES


def read_data(file, labeled):  # pylint: disable=too-many-statements
    '''
    description

    :param file:        file we're reading in with the data (.csv)

    :return             a 'songs' dictionary
                        key: track_id
                        value(s): a 'data' dictionary, which contains:
                            metadata,
                            label,
                            features,
                            genres
    '''

    songs = {}

    with open(file, 'r') as file_data:

        reader = csv.DictReader(file_data)

        for song in tqdm_notebook(reader, desc='Reading data from .csv...'):

            metadata, data = {}, {}
            genres = []
            track_id = song['track_id']

            features = {}

            metadata['artist_id'] = song['artist_id']
            metadata['artist_name'] = song['artist_name']
            metadata['track_name'] = song['track_name']

            # check if genre field has multiple genres or just one
            if song['artist_genres']:
                all_genres = song['artist_genres'].split(',')
                for genre in all_genres:
                    if genre[0] == '"':
                        genres.append(genre[1:])
                    elif genre[-1] == '"':
                        genres.append(genre[:-1])
                    else:
                        genres.append(genre)

            for feature in FLOAT_FEATURE_LIST:
                features[feature] = float(song[feature])

            # add metadata, features, genres, and label to data
            data['metadata'] = metadata
            data['features'] = features
            data['genres'] = genres
            if (labeled):
                data['label'] = int(song['label'])

            # add data to songs by track_id
            songs[track_id] = data

    return songs


def is_float(string):
    '''
    description

    :param string:      the string we're testing

    :return             True if the string is a float, False otherwise
    '''
    try:
        float(string)
        return True
    except ValueError:
        return False


def read_genres(file):
    '''
    description

    :param file:        file we're reading in with the data (.csv)

    :return             a 'genres' dictionary
                        key: genre
                        value: unique genre_id
    '''
    genre_mapping = {}
    genre_id = 0

    with open(file, 'r') as file_data:

        reader = csv.DictReader(file_data)
        genres = []

        for song in tqdm_notebook(reader, desc='Reading data from .csv...'):

            if song['artist_genres']:
                all_genres = song['artist_genres'].split(',')
                for genre in all_genres:
                    if genre[0] == '"':
                        genres.append(genre[1:])
                    elif genre[-1] == '"':
                        genres.append(genre[:-1])
                    else:
                        genres.append(genre)

                # for genres in the genres list, ensure that the genre is not a
                # float, create unique genre_id
                for genre in genres:
                    if genre not in genre_mapping:
                        genre_mapping[genre] = genre_id
                        genre_id += 1

    return genre_mapping


def filter_features(data, discard):
    '''
    Filters out features from data. Does not modify passed-in object (creates a copy).

    :param data:        dict of data, same format as song_data
    :param discard:     feature names to discard

    :return             copy of data, with filtered features
    '''
    out = dict(data)

    for id_ in data:
        out[id_] = deepcopy(out[id_])
        out[id_]['features'] = {k: v for k, v in out[id_]
                                ['features'].items() if k not in discard}

    return FrozenMap(out)


def split_data(data, proportion):
    '''
    Splits data into training and validation sets for simple classification.

    :param data:    complete labeled data
    :param p:       proportion of data to use for validation

    :return         train_data, validation_data
    '''
    validation_ids = rand_sample(list(data), int(len(data) * proportion))

    validation_data = {k: data[k] for k in validation_ids}
    train_data = {k: v for k, v in data.items() if k not in validation_data}

    return FrozenMap(train_data), FrozenMap(validation_data)


def test_cluster_size(data, max_cluster):
    '''
    description

    :param data:        the data, obtained from read_data()
    :param max_cluster: the max number of clusters to km.inertia_ on

    :return             None (prints km_inertia for each number of clusters)
    '''
    train_data = {k: v for k, v in data.items()}
    x_train = np.array([list(x['features'].values())
                        for x in train_data.values()])

    for i in range(2, max_cluster + 1):
        km_classifier = KMeans(
            i,
            init='random',
            max_iter=300,
            random_state=0,
            n_init=30)
        km_classifier.fit(x_train)
        print(i, km_classifier.inertia_)


def get_kmeans_clusters(data, n_clusters, songs_by_cluster={}):  # pylint: disable=dangerous-default-value
    '''
    Assigns songs in data to :n_clusters: unique clusters.

    FUTURE: Refactor feature engineering such that there data
            isn't copied wholesale. E.g., implement multiple
            kinds of features or better feature filtering.
            This will require changes to Classifier.

    :param data:        the data, obtained from read_data()
    :param n_clusters:  the number of clusters used in K-means

    :return             deep copy of data with cluster property and clusters as boolean features
                        dict, n_clusters:cluster:ids_of_songs_in_cluster
    '''

    data = dict(data)

    if n_clusters in songs_by_cluster.keys():
        raise ValueError(str(n_clusters) + ' clusters already computed')

    songs_by_cluster[n_clusters] = {i: set() for i in range(n_clusters)}

    train_data, track_ids = {k: v for k, v in data.items()}, list(data)

    x_train = np.array([list(x['features'].values())
                        for x in train_data.values()])
    y_train = [x['label'] for x in train_data.values()]

    km_classifier = KMeans(
        n_clusters,
        init='random',
        max_iter=300,
        random_state=0,
        n_init=30)
    km_classifier.fit(x_train)

    cluster_map = pd.DataFrame()

    cluster_map['data'], cluster_map['cluster'] = x_train.tolist(
    ), km_classifier.labels_
    cluster_map['label'], cluster_map['track_id'] = y_train, track_ids

    for track_id in track_ids:

        data[track_id] = deepcopy(data[track_id])

        cluster = cluster_map[cluster_map['track_id'] == track_id]['cluster'].tolist()[
            0]

        songs_by_cluster[n_clusters][cluster].add(track_id)

        data[track_id]['cluster'] = cluster
        for i in range(n_clusters):
            data[track_id]['features']['c' + str(i)] = 1 if cluster == i else 0

    return FrozenMap(data), FrozenMap(songs_by_cluster)


def get_xy(song_data, ids=None):
    '''
    get x and y values from song_data
    '''
    if ids:
        song_data = {k: song_data[k] for k in ids}
    return [x['features'] for x in song_data.values()], [x['label']
                                                         for x in song_data.values()]


def get_x(song_data, ids=None):
    '''
    get x values from song_data
    '''
    if ids:
        song_data = {k: song_data[k] for k in ids}
    return [x['features'] for x in song_data.values()]


def get_features_and_id(song_data):
    '''
    get x values + song ID
    '''
    features = []
    song_ids = []

    for song_id, data in song_data.items():
        features.append(data['features'])
        song_ids.append(song_id)

    return features, song_ids
    # return [x['features'] for x in song_data.values()], [key for key in
    # song_data.keys()]


def scale_data(data):
    '''
    scale specific data values in dataframe
    '''
    new_df = data.copy(deep=True)
    sclr = MinMaxScaler()
    new_df[SCALABLE_VALUES] = sclr.fit_transform(new_df[SCALABLE_VALUES])
    return new_df
