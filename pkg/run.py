'''
run
'''

import sklearn #pylint: disable=unused-import
from sklearn.feature_extraction import DictVectorizer

from .pickclassifier import Classifier
from .wrangling import split_data, get_x
from .constants import ONLINE_ALGS, AL_STRATS, NUM_CLUSTERS


def get_default_vectorizer(song_data_x):
    '''
    get default vectorizer
    '''
    vect = DictVectorizer(sort=True)
    return vect.fit(song_data_x)

def get_experiment_split(default_data, clustered_data, p_value=0.2):
    '''
    get experiment split
    '''
    # if discard: data = filter_features(data, discard)

    default_trd, default_vd = split_data(default_data, p_value)
    return default_trd, default_vd, {
        k: clustered_data[k] for k in default_trd}, {
            k: clustered_data[k] for k in default_vd}


def get_train_clusters(clustered_train_data, all_clusters):
    '''
    get train clusters
    '''
    return {c: [_id for _id in all_clusters[c]
                if _id in clustered_train_data] for c in all_clusters}


def run_active_experiment(  # pylint: disable=too-many-arguments
        default_song_data,
        train_data,
        validation_data,
        iterations,
        alg,
        init_n,
        strat='random',
        clusters=None):
    '''
    Trains :alg: on song_data (with or without filtering features), splits into train/validation,
    and returns accuracy on validation data.
    '''
    default_song_data_x = get_x(default_song_data)
    default_vectorizer = get_default_vectorizer(default_song_data_x)

    clf = Classifier(
        alg,
        train_data,
        default_vectorizer,
        True,
        active_init_n=init_n,
        al_strat=strat,
        clusters=clusters
    )
    accuracies = [clf.validate(validation_data)]

    for i in range(iterations):  # pylint: disable=unused-variable
        clf.active_learn()
        accuracies.append(clf.validate(validation_data))

    return accuracies


def run_active_suite(
        default_song_data,
        train_data,
        validation_data,
        algos,
        strats):
    '''
    run active suite
    '''
    results = {}

    for alg in algos:

        results[alg] = {}

        for strat in strats:
            init_n = 5
            accs = run_active_experiment(
                default_song_data,
                train_data,
                validation_data,
                100,
                alg,
                init_n,
                strat=strat)
            print(len(accs))
            print(alg, 'accs w/', strat)
            for i in range(len((accs))):
                print(int(accs[i] * 10000) / 100, end='\t')
            print()

            results[alg][strat] = accs

    return results


def run_clusters_suite(
        default_song_data,
        train_data,
        validation_data,
        train_clusters):
    '''
    run clusters suite
    '''

    results = {}

    for alg in ONLINE_ALGS:

        results[alg] = {}

        for strat in AL_STRATS:

            accs = run_active_experiment(
                default_song_data,
                train_data,
                validation_data,
                100 // NUM_CLUSTERS,
                alg,
                len(train_clusters),
                strat=strat,
                clusters=train_clusters)
            print(alg, 'accs w/', strat)
            for i in range(len(accs)):
                print(int(accs[i] * 10000) / 100, end='\t')
            print()

            results[alg][strat] = accs

    return results


def run_benchmark(
        alg,
        default_song_data,
        train_data,
        validation_data,
        discard=None): #pylint: disable=unused-argument
    '''
    Trains :alg: on song_data (with or without filtering features), splits into train/validation,
    and returns accuracy on validation data.
    '''
    default_song_data_x = get_x(default_song_data)
    default_vectorizer = get_default_vectorizer(default_song_data_x)

    clf = Classifier(alg, train_data, default_vectorizer, False)
    clf.fit()

    return clf.validate(validation_data)
