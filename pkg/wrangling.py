'''
data wrangling
'''

from random import sample as rand_sample
from copy import deepcopy

import numpy as np
import pandas as pd  # pylint: disable=import-error

from sklearn.cluster import KMeans
from maps import FrozenMap  # pylint: disable=import-error


def read_data(file):  # pylint: disable=too-many-statements
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

        for song in file_data:

            line = song.split(',')

            # ignore first line + ensure that label exists
            if line[0] != 'label' and line[0]:
                label = int(line[0])
                metadata, data = {}, {}
                genres = []
                track_id = line[1]

                features = {}
                features['artist_popularity'] = line[4]
                features['artist_followers'] = line[5]

                metadata['artist_id'] = line[2]
                metadata['artist_name'] = line[3]

                # check if genre field has multiple genres or just one
                if line[6]:
                    if '"' in line[6]:
                        genres.append(line[6][1:])
                    else:
                        genres.append(line[6])

                count = 0

                # if '"' present or next value is a string only containing alphabets,
                # then add to genres list. set count to i+1 when you reach last
                # genre
                for i in range(7, len(line)):
                    if '"' in line[i]:
                        genres.append(line[i][:-1])
                        count = i + 1
                        break
                    if line[i].isalpha():
                        genres.append(line[i])

                # single or no genres, get all other features
                if count == 0:
                    features['instrumentalness'] = float(line[7])
                    features['duration_ms'] = float(line[8])
                    features['time_signature'] = float(line[9])
                    features['acousticness'] = float(line[10])
                    features['speechiness'] = float(line[11])
                    features['energy'] = float(line[12])
                    features['loudness'] = float(line[13])
                    features['tempo'] = float(line[14])
                    features['key'] = float(line[15])
                    features['valence'] = float(line[16])
                    features['danceability'] = float(line[17])
                    features['liveness'] = float(line[18][:-1])

                # multiple genres, get all other features using count
                else:
                    features['instrumentalness'] = float(line[count])
                    features['duration_ms'] = float(line[count + 1])
                    features['time_signature'] = float(line[count + 2])
                    features['acousticness'] = float(line[count + 3])
                    features['speechiness'] = float(line[count + 4])
                    features['energy'] = float(line[count + 5])
                    features['loudness'] = float(line[count + 6])
                    features['tempo'] = float(line[count + 7])
                    features['key'] = float(line[count + 8])
                    features['valence'] = float(line[count + 9])
                    features['danceability'] = float(line[count + 10])
                    features['liveness'] = float(line[count + 11][:-1])

                # add metadata, features, genres, and label to data
                data['metadata'] = metadata
                data['features'] = features
                data['genres'] = genres
                data['label'] = label

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

        for song in file_data:

            line = song.split(',')

            if line[0]:

                genres = []

                # check if genres field is empty or contains one/multiple
                # genres
                if line[6]:
                    if '"' in line[6]:
                        genres.append(line[6][1:])
                    else:
                        genres.append(line[6])

                count = 0  # pylint: disable=unused-variable

                # account for multiple genres
                for i in range(7, len(line)):
                    if '"' in line[i]:
                        genres.append(line[i][:-1])
                        count = i + 1
                        break
                    if line[i].isalpha():
                        genres.append(line[i])

                # for genres in the genres list, ensure that the genre is not a
                # float, create unique genre_id
                for genre in genres:
                    if (genre not in genre_mapping) and (
                            '"' not in genre) and (not is_float(genre)):
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
