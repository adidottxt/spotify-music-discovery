'''
classify songs given a playlist / final function
'''

from maps import FrozenMap # pylint: disable=import-error

from .spotipy import get_playlist_data, get_dataframe, login_to_spotify
from .config import CLIENT_ID, CLIENT_USERNAME, CLIENT_SECRET
from .wrangling import read_data, scale_data, get_features_and_id

SPOTIPY_OBJECT = login_to_spotify(CLIENT_USERNAME, CLIENT_ID, CLIENT_SECRET)
PLAYLISTS = SPOTIPY_OBJECT.user_playlists(CLIENT_USERNAME)


def classify_playlist(playlist_name, classifier):
    '''
    take in playlist, output songs they'd like using classifier
    '''
    # get playlist data
    playlist_data = get_playlist_data(
        SPOTIPY_OBJECT,
        CLIENT_USERNAME,
        PLAYLISTS,
        playlist_name)

    # featurize playlist song data
    playlist_df = get_dataframe(SPOTIPY_OBJECT, playlist_data, -1)

    scaled_playlist_df = scale_data(playlist_df)
    scaled_playlist_df.to_csv(
        'data/{}.csv'.format(playlist_name),
        encoding='utf-8')

    playlist_data_dict = FrozenMap(
        read_data('data/{}.csv'.format(playlist_name), False))
    playlist_features, playlist_song_ids = get_features_and_id(
        playlist_data_dict)

    # run classifier on playlist songs
    results = classifier.predict_all(playlist_features)
    liked_songs = [playlist_song_ids[i]
                   for i in range(len(results)) if results[i] == 1]

    # get songs they'd like based on song ID
    if not liked_songs:
        print(
            "The given classifier thinks you wouldn't like any songs in \
            the given playlist.")
        return

    print("The given classifier thinks you'd like the following from the given playlist:\n")

    for song in liked_songs:
        print(playlist_data_dict[song]['metadata']['track_name'])

    print()
    return
