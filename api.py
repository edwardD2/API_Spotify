# import libraries
import requests
import pandas as pd
import math

# keys
CLIENT_ID = '**********'
CLIENT_SECRET = '**********'

# API call procedures
AUTH_URL = 'https://accounts.spotify.com/api/token'

# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# convert the response to JSON
auth_response_data = auth_response.json()

# save the access token
access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

# base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'


# change duration_ms to 'mm:ss' format
def convert_ms(duration_ms):
    m = math.floor((duration_ms / 1000) / 60)
    s = math.floor((duration_ms / 1000) % 60)
    if s < 10:
        s = '0' + str(s)
    if m < 10:
        m = '0' + str(m)
    duration_ms = str(m) + ':' + str(s)
    return duration_ms


# collect energy, loudness, tempo, valence, danceability, and duration
def get_track_details(track_id):
    r_track = requests.get(BASE_URL + 'audio-features/' + track_id, headers=headers)
    r_track = r_track.json()

    energy = r_track['energy']
    loudness = r_track['loudness']
    tempo = r_track['tempo']
    valence = r_track['valence']
    danceability = r_track['danceability']
    duration_ms = r_track['duration_ms']

    return energy, loudness, tempo, valence, danceability, convert_ms(duration_ms), duration_ms


# get tracks from playlist
def get_tracks(df_tracks, df_artists, j, playlist_id):
    offset = j * 100
    r2 = requests.get(BASE_URL + 'playlists/' + playlist_id + '/tracks?offset=' + str(offset), headers=headers)
    r2 = r2.json()

    artist_ids = []
    total_duration_ms = 0

    # iterate through each track
    for track in r2['items']:
        while True:
            try:
                track_id = track['track']['id']
                track_title = track['track']['name']
                # print(track_title)
                track_artists = track['track']['artists']
                artist_list = []
                for artist in track_artists:
                    artist_list.append(artist['name'])
                    artist_id = artist['id']
                    if artist_id not in artist_ids:
                        artist_ids.append(artist_id)
                track_artists = ", ".join(artist_list)
                track_album = track['track']['album']['name']
                track_release_date = track['track']['album']['release_date']
                energy, loudness, tempo, valence, danceability, duration, duration_ms = get_track_details(track_id)
                df_tracks = df_tracks.append({'track_id': track_id, 'track_title': track_title,
                                              'track_artists': track_artists, 'track_album': track_album,
                                              'track_release_date': track_release_date, 'energy': energy,
                                              'loudness': loudness, 'tempo': tempo, 'valence': valence,
                                              'danceability': danceability, 'duration': duration}, ignore_index=True)
                total_duration_ms += duration_ms
            except KeyError:
                continue
            break

    for artist_id in artist_ids:
        while True:
            try:
                r3 = requests.get(BASE_URL + 'artists/' + artist_id, headers=headers)
                r3 = r3.json()
                artist_name = r3['name']
                genres = r3['genres']
                popularity = r3['popularity']
                df_artists = df_artists.append({'artist_id': artist_id, 'artist_name': artist_name,
                                                'genres': genres, 'popularity': popularity}, ignore_index=True)
            except KeyError:
                continue
            break

    return df_tracks, df_artists, total_duration_ms


# main
username = 'dedwur'
r = requests.get(BASE_URL + 'users/' + username + '/playlists', headers=headers)
r = r.json()

columns_tracks = ['track_id', 'track_title', 'track_artists', 'track_album', 'track_release_date', 'energy',
                  'loudness', 'tempo', 'valence', 'danceability', 'duration', 'playlist']
columns_artists = ['artist_id', 'artist_name', 'genres', 'popularity']
columns_playlists = ['playlist_id', 'playlist_name', 'avg_energy', 'avg_loudness', 'avg_tempo', 'avg_valence',
                     'avg_danceability', 'top_genres', 'total_tracks', 'total_duration']

df_all_tracks = pd.DataFrame(columns=columns_tracks)
df_all_artists = pd.DataFrame(columns=columns_artists)
df_playlists = pd.DataFrame(columns=columns_playlists)

all_top_genres = []

# iterate through every playlist
for i in range(0, len(r['items'])):
    playlist_name = r['items'][i]['name']
    if playlist_name == 'K':
        print('Skipping playlist \'' + playlist_name + '\'.')
        continue
    print('Grabbing tracks from \'' + playlist_name + '\'...')
    playlist_id = r['items'][i]['id']
    total_tracks = r['items'][i]['tracks']['total']
    total_duration_ms = 0

    # iterate through every track in playlist per 100
    for j in range(0, math.floor(total_tracks / 100) + 1):
        df_track = pd.DataFrame(columns=columns_tracks)
        df_artists = pd.DataFrame(columns=columns_artists)

        # add df values
        df_track, df_artists, duration_ms = get_tracks(df_track, df_artists, j, playlist_id)
        if playlist_name == '떡볶이':
            df_track['playlist'] = 'tteokbokki'
        else:
            df_track['playlist'] = playlist_name

        # combine df values into one collective df
        df_all_tracks = df_all_tracks.append(df_track, ignore_index=True)
        df_all_artists = df_all_artists.append(df_artists, ignore_index=True)

        total_duration_ms += duration_ms

    # designated 'predictions' playlist for model application
    # 'predictions' needs to be the first playlist
    if playlist_name == 'predictions':
        df_all_tracks.to_csv('csv files/predictions.csv', index=False)
        print('predictions.csv has been created.')
        df_all_tracks = pd.DataFrame(columns=columns_tracks)
        df_all_artists = pd.DataFrame(columns=columns_artists)
        continue

    avg_energy = round(df_all_tracks['energy'].mean(), 4)
    avg_loudness = round(df_all_tracks['loudness'].mean(), 4)
    avg_tempo = round(df_all_tracks['tempo'].mean(), 4)
    avg_valence = round(df_all_tracks['valence'].mean(), 4)
    avg_danceability = round(df_all_tracks['danceability'].mean(), 4)

    genres = {}
    while True:
        try:
            for id in df_artists['artist_id']:
                r4 = requests.get(BASE_URL + 'artists/' + id, headers=headers)
                r4 = r4.json()
                for genre in r4['genres']:
                    if genre in genres.keys():
                        genres[genre] += 1
                    else:
                        genres[genre] = 1
        except KeyError:
            continue
        break

    top_genres = sorted(genres, key=genres.get, reverse=True)[:5]

    for genre in top_genres:
        if genre not in all_top_genres:
            all_top_genres.append(genre)

    hh = math.floor((total_duration_ms / 1000) / 3600)
    mmss = convert_ms(total_duration_ms - (hh * 3600000))
    if hh < 10:
        hh = '0' + str(hh)
    total_duration = str(hh) + ':' + mmss

    df_playlists = df_playlists.append({'playlist_id': playlist_id, 'playlist_name': playlist_name,
                                        'avg_energy': avg_energy, 'avg_loudness': avg_loudness,
                                        'avg_tempo': avg_tempo, 'avg_valence': avg_valence,
                                        'avg_danceability': avg_danceability, 'top_genres': top_genres,
                                        'total_tracks': total_tracks, 'total_duration': total_duration
                                        }, ignore_index=True)

df_all_tracks.to_csv('csv files/tracks.csv', index=False)
print('tracks.csv has been created.')
df_all_artists.drop_duplicates(subset='artist_id').to_csv('csv files/artists.csv', index=False)
print('artists.csv has been created.')
df_playlists.to_csv('csv files/playlists.csv', index=False)
print('playlists.csv has been created.')
atg = {'all_top_genres': all_top_genres}
pd.DataFrame(atg).to_csv('csv files/genres.csv', index=False)
print('genres.csv has been created.')
