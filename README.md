# 🎧 Spotify Music Discovery

Create your own Spotify recommendation algorithm. All you need is Python 3, 
Jupyter Notebook, and a Spotify account. 

## Setup

1. Clone/download the repository.
2. Ensure that you have [Python 3](https://www.python.org/downloads/) and 
[Jupyter Notebook](https://jupyter.org/install) installed.
3. Navigate to `/spotify-music-discovery` and run `pip install -r requirements.txt`.
4. Create the file `spotify-music-discovery/pkg/config.py` as below, using 
your [Spotify client information](https://developer.spotify.com/documentation/general/guides/app-settings/#register-your-app).
  ```python
  CLIENT_ID = 'your Spotify client id here'
  CLIENT_SECRET = 'your Spotify client secret here'
  CLIENT_USERNAME = 'your Spotify username here'
  ```
5. To help avoid any OAuth errors that might occur, open your Spotify 
application [here](https://developer.spotify.com/dashboard/applications), and 
set up your Redirect URIs as follows:
  - `Edit Settings`
    - Add `http://localhost:8888/callback/` under "Redirect URIs"

6. When prompted in `get_spotify_data`, copy and paste the link you are 
redirected to in the input box that should pop up after running the first 
cell (even if the link throws a "localhost redirected you too many times" 
error).

## Usage

- Start Jupyter Notebook in the `/spotify-music-discovery` directory.
- Run through the notebooks in sequence, following the instructions in each:
  1. `get_spotify_data`
      - Downloads and parses song data from the training playlists you specify.
  2. `train`
      - Trains classifiers using the training data and saves the best one.
        - Specifically, it pickles the classifier object and writes it to the `./classifiers` directory.
  3. `predict`
      - Predicts which songs you like from a playlist you specify.

#### Spotify URIs

To download playlists, you will have to specify their Spotify URIs. You can get 
Spotify URIs from the Spotify app, as follows:
- Right-click on a playlist
  - `Share`
    - `Copy Spotify URI`

## Background

This repo was originally created for a final project in an applied machine
learning course at the University of Pennsylvania. For more detail, including 
the algorithms used, see the
[project report](music_discovery_using_active_learning.pdf).
