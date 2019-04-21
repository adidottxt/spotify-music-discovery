# spotify-music-discovery

### Setup
Before running any of the above code or notebooks, set up a `config.py` file in the repository 
with the following information from [Spotify](https://developer.spotify.com/documentation/general/guides/app-settings/#register-your-app):

```python
CLIENT_ID = 'your Spotify client id here'
CLIENT_SECRET = 'your Spotify client secret here'
CLIENT_USERNAME = 'your Spotify username here'
LIKES_PLAYLIST = 'your "likes" Spotify playlist URI/ID here' # (i.e. positive examples)
DISLIKES_PLAYLIST = 'your "dislikes" Spotify playlist URI/ID here' # (i.e. negative examples)
```
