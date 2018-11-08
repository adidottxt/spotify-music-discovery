'''
This handles logging in to Spotify and returning a Spotipy object
'''

import os

import spotipy
import spotipy.util as util

import config


def login_to_spotify():
    '''
    handles logging in to spotify and returning a spotipy object
    '''

    os.environ['SPOTIPY_CLIENT_ID'] = config.CLIENT_ID
    os.environ['SPOTIPY_CLIENT_SECRET'] = config.CLIENT_SECRET
    os.environ['SPOTIPY_REDIRECT_URI'] = 'http://localhost:8888/callback/'

    username = ''

    scope = 'user-library-read playlist-read-private user-top-read'

    token = util.prompt_for_user_token(
        username,
        scope
    )

    spotipy_wrapper = spotipy.Spotify(
        auth=token
    )

    if token:
        spotipy_wrapper = spotipy.Spotify(auth=token)
        print("Authentication done")
        return spotipy_wrapper

    print("Can't get token for", username)
    return None
