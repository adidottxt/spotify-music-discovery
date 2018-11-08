'''
This is where the program is actually run
'''

from login import login_to_spotify

def show_tracks(t_tracks):
    '''
    test
    '''
    for i, item in enumerate(t_tracks['items']):
        t_track = item['track']
        print("%d %32.32s %s" % (i, t_track['artists'][0]['name'], t_track['name']))

SPOTIPY_OBJECT = login_to_spotify()

def print_all_songs(username):
    '''
    print all songs in all playlists
    '''
    playlists = SPOTIPY_OBJECT.user_playlists(username)

    for playlist in playlists['items']:
        if playlist['owner']['id'] == username:
            print(playlist['name'])
            print('  total tracks', playlist['tracks']['total'])

            results = SPOTIPY_OBJECT.user_playlist(
                username,
                playlist['id'],
                fields="tracks,next"
            )

            tracks = results['tracks']
            show_tracks(tracks)

            while tracks['next']:
                tracks = SPOTIPY_OBJECT.next(tracks)
                show_tracks(tracks)
