'''
spotify functions
'''

import os
import spotipy
import spotipy.util as util
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm_notebook, trange
import pandas as pd

from .constants import FINAL_COLUMNS, TRACK_FEATURES, LIKE, DISLIKE, NO_LABEL_FINAL_COLUMNS


def login_to_spotify(username, client_id, client_secret):
    '''
    This handles logging in to Spotify and returning a Spotipy object
    to be used to gather our data
    '''

    os.environ['SPOTIPY_CLIENT_ID'] = client_id
    os.environ['SPOTIPY_CLIENT_SECRET'] = client_secret
    os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/callback/'

    scope = 'user-library-read playlist-read-private user-top-read'

    token = util.prompt_for_user_token(
        username,
        scope,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri='http://localhost:8888/callback/',
    )

    client_credentials_manager = SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret)
    spotify = spotipy.Spotify(
        client_credentials_manager=client_credentials_manager)

    return spotify


def get_playlist_data(spotipy_object, username, playlists, playlist_name):
    '''
    This function takes in a user's playlists and a playlist name,
    and downloads all song information for the given playlist name.
    '''

    for playlist in tqdm_notebook(
            playlists['items'],
            desc='Finding playlist "{}"...'.format(playlist_name)):
        if playlist['name'] == playlist_name:
            playlist_id = playlist['id']

    if not playlist_id:
        print("playlist not found!")
        return None

    playlist_data = spotipy_object.user_playlist(
        username,
        playlist_id,
    )

    playlist_tracks = playlist_data["tracks"]
    playlist_songs = playlist_tracks["items"]

    while playlist_tracks['next']:

        playlist_tracks = spotipy_object.next(playlist_tracks)
        for item in playlist_tracks['items']:
            playlist_songs.append(item)

    return playlist_songs


def get_dataframe(spotipy_object, playlist_data, label):
    '''
    get dataframe from data
    '''
    track_ids = []
    track_names = {}
    for track in tqdm_notebook(playlist_data, desc='Saving playlist data...'):
        if track['track']['id']:
            track_names[track['track']['id']] = track['track']['name']
            track_ids.append(track['track']['id'])

    track_features_list = []

    for i in tqdm_notebook(range(0, len(track_ids), 50),
                           desc='Downloading audio features...'):
        track_features_list.extend(
            spotipy_object.audio_features(tracks=track_ids[i:i + 50]))

    track_features = {}

    for i in tqdm_notebook(
            range(
                len(track_ids)),
            desc='Parsing track features...'):
        track_features[track_ids[i]] = track_features_list[i]

    artist_ids = {}
    artist_genres = {}
    artist_names = {}
    artist_popularity = {}
    artist_followers = {}

    for record in tqdm_notebook(track_ids, desc="Parsing artist data..."):

        artist_id = spotipy_object.track(record)['artists'][0]['id']
        artist_info = spotipy_object.artist(artist_id)
        artist_ids[record] = artist_id

        artist_genres[artist_id] = ','.join(artist_info['genres'])
        artist_names[artist_id] = artist_info['name']
        artist_popularity[artist_id] = artist_info['popularity']
        artist_followers[artist_id] = artist_info['followers']['total']

    track_features_df = pd.DataFrame.from_dict(
        track_features,
        orient='index').reset_index().rename(
        columns={
            'id': 'track_id'}).drop(
                columns=['index'])[TRACK_FEATURES]

    if label in (LIKE, DISLIKE):
        final_df = pd.DataFrame(columns=FINAL_COLUMNS)
    else:
        final_df = pd.DataFrame(columns=NO_LABEL_FINAL_COLUMNS)

    track_energy = track_features_df[['track_id', 'energy']].set_index(
        'track_id').to_dict()['energy']
    track_tempo = track_features_df[['track_id', 'tempo']].set_index(
        'track_id').to_dict()['tempo']
    track_time_signature = track_features_df[['track_id', 'time_signature']].set_index(
        'track_id').to_dict()['time_signature']
    track_valence = track_features_df[['track_id', 'valence']].set_index(
        'track_id').to_dict()['valence']
    track_duration_ms = track_features_df[['track_id', 'duration_ms']].set_index(
        'track_id').to_dict()['duration_ms']
    track_key = track_features_df[['track_id', 'key']].set_index(
        'track_id').to_dict()['key']
    track_liveness = track_features_df[['track_id', 'liveness']].set_index(
        'track_id').to_dict()['liveness']
    track_speechiness = track_features_df[['track_id', 'speechiness']].set_index(
        'track_id').to_dict()['speechiness']
    track_danceability = track_features_df[['track_id', 'danceability']].set_index(
        'track_id').to_dict()['danceability']
    track_loudness = track_features_df[['track_id', 'loudness']].set_index(
        'track_id').to_dict()['loudness']
    track_instrumentalness = track_features_df[['track_id', 'instrumentalness']].set_index(
        'track_id').to_dict()['instrumentalness']
    track_acousticness = track_features_df[['track_id', 'acousticness']].set_index(
        'track_id').to_dict()['acousticness']

    for i in tqdm_notebook(
            range(
                len(track_ids)),
            desc="Building final dataframe..."):

        track_id = track_ids[i]
        track_name = track_names[track_id]
        data = []
        artist_id = artist_ids[track_id]

        if label in (LIKE, DISLIKE):
            data.extend(
                (
                    label,
                    track_id,
                    artist_id,
                    artist_names[artist_id],
                    artist_popularity[artist_id],
                    artist_followers[artist_id],
                    artist_genres[artist_id],
                    track_instrumentalness[track_id],
                    track_duration_ms[track_id],
                    track_time_signature[track_id],
                    track_acousticness[track_id],
                    track_speechiness[track_id],
                    track_energy[track_id],
                    track_loudness[track_id],
                    track_tempo[track_id],
                    track_key[track_id],
                    track_valence[track_id],
                    track_danceability[track_id],
                    track_liveness[track_id],
                    track_name
                )
            )
        else:
            data.extend(
                (
                    track_id,
                    artist_id,
                    artist_names[artist_id],
                    artist_popularity[artist_id],
                    artist_followers[artist_id],
                    artist_genres[artist_id],
                    track_instrumentalness[track_id],
                    track_duration_ms[track_id],
                    track_time_signature[track_id],
                    track_acousticness[track_id],
                    track_speechiness[track_id],
                    track_energy[track_id],
                    track_loudness[track_id],
                    track_tempo[track_id],
                    track_key[track_id],
                    track_valence[track_id],
                    track_danceability[track_id],
                    track_liveness[track_id],
                    track_name
                )
            )

        final_df.loc[i] = data

    return final_df
