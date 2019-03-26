'''
Classifier object
'''
from random import sample as rand_sample

from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
import numpy as np

from .constants import SUPPORTED_ALGS, AL_STRATS, ONLINE_ALGS, CLASSES
from .wrangling import get_xy, get_x


def get_default_learner(algorithm):
    '''
    get default learner
    '''
    if algorithm == 'svc':
        return SVC(gamma='auto')  # defaults

    if algorithm == 'lsvc':
        return LinearSVC(loss='hinge', penalty='l2')  # defaults

    if algorithm == 'sgd':
        return SGDClassifier(loss='hinge', penalty='l2')  # defaults

    raise ValueError('unknown algorithm: ' + str(algorithm))

class Classifier():  # pylint: disable=too-many-instance-attributes
    '''
    Classifier object
    '''

    def __init__(  # pylint: disable=too-many-arguments
            self,
            algorithm,
            train_data,
            vectorizer,
            is_active,
            active_init_n=None,
            al_strat=None,
            clusters=None
    ):
        '''
        Class providing an interface for learning experiments.

        :param algorithm:       the name of the learning algorithm to use
        :param train_data:      dict
        :param vectorizer:      DictVectorizer, fit on train_data features
        :param is_active:       bool
        :param active_init_n:   number of instances to initialize active learner with
        :param al_strat:        str, the active learning strategy to use

        :return                 Classifier instance
        '''

        if algorithm not in SUPPORTED_ALGS:
            raise ValueError('Unsupported algorithm: ' + str(algorithm))
        if is_active:
            if al_strat not in AL_STRATS:
                raise ValueError('Unsupported AL strategy: ' + str(al_strat))
            if not active_init_n or active_init_n <= 0:
                raise ValueError(
                    'Invalid active_init_n: ' +
                    str(active_init_n))
        if clusters and algorithm not in ONLINE_ALGS:
            raise ValueError(
                'Algorithm must be online if using cluster sampling.')

        self.algorithm = algorithm
        self.train_data = train_data
        self.vectorizer = vectorizer
        self._is_active = is_active
        self._is_online = algorithm in ONLINE_ALGS
        self.strategy = al_strat
        self.active_init_n = active_init_n
        self.learner = get_default_learner(self.algorithm)

        self.clusters = clusters
        self._uses_cluster_sampling = bool(clusters)
        self.num_clusters = len(clusters) if clusters else None

        if not self.is_active():
            self.x_train, self.y_train = get_xy(self.train_data)

        else:
            if not self.uses_cluster_sampling():
                self.unseen_ids = list(self.train_data.keys())
            else:
                self.unseen_ids = {
                    i: list(
                        self.clusters[i]) for i in range(
                            self.num_clusters)}

            self.train_ids = set()

            self.active_learn(clusters=self.active_init_n, init=True)

    def is_active(self):
        '''
        checks if active
        '''
        return self._is_active

    def is_online(self):
        '''
        checks if online
        '''
        return self._is_online

    def uses_cluster_sampling(self):
        '''
        checks if cluster sampling used
        '''
        return self._uses_cluster_sampling

    def transform(self, feature_list):
        '''
        Vectorizes list of feature(s)
        '''
        return self.vectorizer.transform(feature_list)

    def fit(self):
        '''
        fit algo to dataset
        '''
        if not self.is_active():
            self.learner.fit(
                self.vectorizer.transform(
                    self.x_train), self.y_train)
        else:
            self.active_learn()

    def active_learn(self, clusters=1, init=False):
        '''
        active learning done
        '''

        if not self.uses_cluster_sampling():
            sample_ids = self.al_sample(clusters, init)
        else:
            sample_ids = self.al_sample_clusters(clusters, init)

        if not self.is_online():
            sample_x, sample_y = get_xy(self.train_data, ids=self.train_ids)
            self.learner = get_default_learner(self.algorithm)
            self.learner.fit(self.transform(sample_x), sample_y)
        else:
            sample_x, sample_y = get_xy(self.train_data, ids=sample_ids)
            if init:
                self.learner.partial_fit(
                    self.transform(sample_x), sample_y, CLASSES)
            else:
                self.learner.partial_fit(self.transform(sample_x), sample_y)

    def al_sample(self, number, init):
        '''
        active learning sample
        '''

        strategy = self.strategy if not init else 'random'

        if strategy == 'random':

            # Initial samples must contain all labels for offline learners.
            # This ensures that happens.
            if init and not self.is_online():
                sampled_classes = set()
                while len(sampled_classes) != len(CLASSES):
                    sampled_classes = set()
                    sample_ids = rand_sample(self.unseen_ids, number)
                    sampled_classes = set()
                    for id_ in sample_ids:
                        sampled_classes.add(self.train_data[id_]['label'])
            else:
                sample_ids = rand_sample(self.unseen_ids, number)

            for song_id in sample_ids:
                self.unseen_ids.remove(song_id)

        elif strategy == 'uncertainty':
            unseen_x = get_x(
                self.train_data, ids=self.unseen_ids)
            unseen_x_scores = self.learner.decision_function(
                self.transform(unseen_x))

            # get index of smallest absolute value (probability or distance from decision boundary)
            # from unseen_x_scores, pop the corresponding entry from
            # self.unseen_ids
            sample_ids = [
                self.unseen_ids.pop(
                    np.argmin(
                        np.abs(unseen_x_scores)))]

        self.train_ids.update(sample_ids)

        return sample_ids

    def al_sample_clusters(self, num_clusters, init):
        '''
        active learning with clusters
        '''

        strategy = self.strategy if not init else 'random'

        sample_ids = []

        for i in range(self.num_clusters):

            if strategy == 'random':
                new_sample_ids = rand_sample(self.unseen_ids[i], num_clusters)
                for song_id in new_sample_ids:
                    self.unseen_ids[i].remove(song_id)
                sample_ids += new_sample_ids

            elif strategy == 'uncertainty':
                unseen_x = get_x(
                    self.train_data, ids=self.unseen_ids[i])
                unseen_x_scores = self.learner.decision_function(
                    self.transform(unseen_x))

                # get index of smallest absolute value (probability or
                # distance from decision boundary)
                # from unseen_x_scores, pop the corresponding entry from
                # self.unseen_ids
                sample_ids.append(
                    self.unseen_ids[i].pop(
                        np.argmin(
                            np.abs(unseen_x_scores))))

        self.train_ids.update(sample_ids)

        return sample_ids

    def predict(self, value, learner=None):  # pylint: disable=unused-argument
        '''
        predict using classifier
        '''
        return self.learner.predict(value)[0]

    def validate(self, validation_data):
        '''
        Predicts on all instances in validation_data. Returns accuracy.
        '''
        x_test, y_test = get_xy(validation_data)

        correct = 0
        for i in range(len(validation_data)):
            if self.predict(self.transform([x_test[i]])) == y_test[i]:
                correct += 1

        return correct / len(validation_data)
