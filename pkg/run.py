'''
run
'''

import sklearn  # pylint: disable=unused-import
from sklearn.feature_extraction import DictVectorizer

from .spotifyclassifier import Classifier
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
        discard=None):  # pylint: disable=unused-argument
    '''
    Trains :alg: on song_data (with or without filtering features), splits into train/validation,
    and returns accuracy on validation data.
    '''
    default_song_data_x = get_x(default_song_data)
    default_vectorizer = get_default_vectorizer(default_song_data_x)

    clf = Classifier(alg, train_data, default_vectorizer, False)
    clf.fit()

    return clf.validate(validation_data), clf


def get_highest_benchmark( #pylint: disable=too-many-locals
        default_song_data,
        default_training_data,
        default_validation_data,
        clustered_training_data,
        clustered_validation_data):
    '''
    run all benchmarks, get highest, return classifier
    '''

    results = {}

    svc_bench_unclustered, svc_bench_unclustered_clf = run_benchmark(
        'svc',
        default_song_data,
        default_training_data,
        default_validation_data)
    # print(svc_bench_unclustered)
    results['svc_unclustured'] = svc_bench_unclustered, svc_bench_unclustered_clf

    svc_bench_clustered, svc_bench_clustered_clf = run_benchmark(
        'svc',
        default_song_data,
        clustered_training_data,
        clustered_validation_data)
    # print(svc_bench_clustered)
    results['svc_clustured'] = svc_bench_clustered, svc_bench_clustered_clf

    lsvc_bench_unclustered, lsvc_bench_unclustered_clf = run_benchmark(
        'lsvc',
        default_song_data,
        default_training_data,
        default_validation_data)
    # print(lsvc_bench_unclustered)
    results['lsvc_unclustured'] = lsvc_bench_unclustered, lsvc_bench_unclustered_clf

    lsvc_bench_clustered, lsvc_bench_clustered_clf = run_benchmark(
        'lsvc',
        default_song_data,
        clustered_training_data,
        clustered_validation_data)
    # print(lsvc_bench_clustered)
    results['lsvc_clustured'] = lsvc_bench_clustered, lsvc_bench_clustered_clf

    sgd_bench_unclustered, sgd_bench_unclustered_clf = run_benchmark(
        'sgd',
        default_song_data,
        default_training_data,
        default_validation_data)
    # print(sgd_bench_unclustered)
    results['sgd_unclustured'] = sgd_bench_unclustered, sgd_bench_unclustered_clf

    sgd_bench_clustered, sgd_bench_clustered_clf = run_benchmark(
        'sgd',
        default_song_data,
        clustered_training_data,
        clustered_validation_data)
    # print(sgd_bench_clustered)
    results['sgd_clustured'] = sgd_bench_clustered, sgd_bench_clustered_clf

    final_classifier = max(results, key=lambda key: results[key][0])
    # print(final_classifier, '== best classifier')
    return results[final_classifier][1]
