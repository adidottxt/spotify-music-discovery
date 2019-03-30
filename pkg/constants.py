'''
constants
'''

SUPPORTED_ALGS = ['svc', 'lsvc', 'sgd']
ONLINE_ALGS = ['sgd']
AL_STRATS = ['random', 'uncertainty']
CLASSES = [0, 1]
NUM_CLUSTERS = 4
LABEL_LIST = ['Unclustered', 'Clustered']
LIKE = 1
DISLIKE = 0

SCALABLE_VALUES = [
    'duration_ms',
    'key',
    'tempo',
    'loudness',
    'artist_followers',
    'artist_popularity',
    'time_signature'
]


TRACK_FEATURES = [
    'acousticness',
    'track_id',
    'instrumentalness',
    'energy',
    'tempo',
    'time_signature',
    'valence',
    'duration_ms',
    'key',
    'liveness',
    'speechiness',
    'danceability',
    'loudness',
]

FLOAT_FEATURE_LIST = [
    'artist_popularity',
    'artist_followers',
    'instrumentalness',
    'duration_ms',
    'time_signature',
    'acousticness',
    'speechiness',
    'energy',
    'loudness',
    'tempo',
    'key',
    'valence',
    'danceability',
    'liveness'
]

FINAL_COLUMNS = [
    'label',
    'track_id',
    'artist_id',
    'artist_name',
    'artist_popularity',
    'artist_followers',
    'artist_genres',
    'instrumentalness',
    'duration_ms',
    'time_signature',
    'acousticness',
    'speechiness',
    'energy',
    'loudness',
    'tempo',
    'key',
    'valence',
    'danceability',
    'liveness',
    'track_name',
]

NO_LABEL_FINAL_COLUMNS = [
    'track_id',
    'artist_id',
    'artist_name',
    'artist_popularity',
    'artist_followers',
    'artist_genres',
    'instrumentalness',
    'duration_ms',
    'time_signature',
    'acousticness',
    'speechiness',
    'energy',
    'loudness',
    'tempo',
    'key',
    'valence',
    'danceability',
    'liveness',
    'track_name',
]
