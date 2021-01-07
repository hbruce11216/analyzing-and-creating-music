#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:07:18 2020

@author: holdenbruce
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd 

cid = '52d3aec17a9044f0a3b0aac1d83eb356'
secret = '36e79b71874a437589d269c445815bd3'
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)

my_user_id = '1218037846'
disco_party_id = '0l84JP9zEu8NLIGVjHyR1W'

def get_songs_from_playlist(user_id=my_user_id,playlist_id=disco_party_id):
    #user_id = my_user_id
    #playlist_id = disco_party_id
    offset = 0 #initialize offset at 0
    songs = [] #initialize list for holding song names
    items = [] #initialize list for holding song data
    ids = [] #initialize list for holding song ids

    # loop through tracks in playlist and add song data to the items list
    while True:
        content = sp.user_playlist_tracks(user_id,playlist_id,fields=None, limit=100, offset=offset, market=None)
        items += content['items']
        if content['next'] is not None:
            offset += 100
        else:
            break
        
    #add track id and track name to their respective lists
    for song in items:
        ids.append(song['track']['id'])
        songs.append(song['track']['name'])

    #return song names and song ids
    return songs, ids 


def get_audio_features(ids,songs):
    index = 0 #initialize index at 0
    audio_features = [] #initialize list for holding audio_features
    
    #loop through list of ids and append the audio_features for each
    #id to the audio_features list, increment index by 1
    while index < len(ids):
        audio_features += sp.audio_features(ids[index:index+1])
        index += 1
        
    len(audio_features)
    audio_features[0]
    
    features_list = []
    for features in audio_features:
        features_list.append([features['danceability'],
                              features['energy'],
                              features['key'],
                              features['loudness'],
                              features['mode'],
                              features['speechiness'],
                              features['acousticness'],
                              features['instrumentalness'],
                              features['liveness'],
                              features['valence'],
                              features['tempo'],
                              features['type'],
                              features['uri']                                                           
                              ])
    
    
    df = pd.DataFrame(features_list, columns=['danceability',
                                              'energy',
                                              'key',
                                              'loudness',
                                              'mode',
                                              'speechiness',
                                              'acousticness',
                                              'instrumentalness',
                                              'liveness',
                                              'valence',
                                              'tempo',
                                              'type',
                                              'uri'                                              
                                              ])
    
    #create a csv file that holds a pandas dataframe with each audio_feature
    #as a column and every song in the playlist as a row
    #next, we will use this csv to begin out exploratory data analysis 
    #of the type of music in my playlist 
    df.to_csv('{}-{}.csv'.format(my_user_id,disco_party_id), index=False)
    
    
    
    
    
    

import pandas as pd
import numpy as np
disco_party_csv = pd.read_csv('1218037846-0l84JP9zEu8NLIGVjHyR1W.csv', na_values='?').dropna()
disco_party_csv.isna().sum().sum() #no missing values
disco_party_csv.head()
disco_party_csv.columns
disco_party_csv.info()
disco_party_csv.shape
disco_party_csv.describe(include='all')

# sns.pairplot(disco_party_csv,hue='danceability')

column_correlation = disco_party_csv[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
       'type', 'uri']].corr()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='darkgrid')
# %matplotlib inline
plt.figure(figsize=(12,8))
sns.heatmap(column_correlation, annot=True)



# Let’s also check top 10 artists in terms of average energy per 
#song and compare the results with their average acousticness values.
# df[['artists','energy','acousticness']].groupby('artists').mean().sort_values(by='energy', ascending=False)[:10]
disco_party_csv[['danceability','energy','valence','tempo']].mean().sort_values(ascending=False)[:10]

# return top 10 songs with highest danceability
top10_dance = disco_party_csv.nlargest(10,'danceability')
# return top 10 songs with highest energy
top20_energy = disco_party_csv.nlargest(20,'energy')
# return top 10 songs with highest valence
top10_valence = disco_party_csv.nlargest(10,'valence')
# return top 10 songs with highest tempo
top10_tempo = disco_party_csv.nlargest(10,'tempo')


#print the acousticness for the top10_dance list
print(top10_dance['acousticness']) 
#very low accousticness for the most danceable songs...synthetic?

#print the valence for the top10_dance list
print(top10_dance['valence']) 
#generally high valence for most danceable songs...
#valence is all just about having a positive tone to the song

#print the tempo for the top10_energy list
print(top20_energy['tempo'])
print(top20_energy['tempo'].mean()) #mean tempo for top20_energy = 124.0


#i like dancey songs:
disco_party_csv.danceability.mean() #avg is 0.7547
disco_party_csv.danceability.max() #max is 0.941
disco_party_csv.danceability.min() #lowest is 0.528

#i typically don't like accoustic songs
disco_party_csv.acousticness.mean() #avg is 0.0679
disco_party_csv.acousticness.max() #max is 0.76
disco_party_csv.acousticness.min() #lowest is 0.0000111

#i like high energy songs
disco_party_csv.energy.mean() #avg is 0.731
disco_party_csv.energy.max() #max is 0.99
disco_party_csv.energy.min() #lowest is 0..356

#big range in my liking of instrumentalness...insignificant
#loudness also seems to be insignificant
disco_party_csv.instrumentalness.mean() #avg is 0.394
disco_party_csv.instrumentalness.max() #max is 0.954
disco_party_csv.instrumentalness.min() #lowest is 0.0

#i tend to like songs with low speechiness
disco_party_csv.speechiness.mean() #avg is 0.077
disco_party_csv.speechiness.max() #max is 0.471
disco_party_csv.speechiness.min() #lowest is 0.0263

#i tend to like songs with low speechiness
disco_party_csv.liveness.mean() #avg is 0.077
disco_party_csv.liveness.max() #max is 0.471
disco_party_csv.liveness.min() #lowest is 0.0263




## b
# Logistic Regression
import statsmodels.api as sm
disco_party_csv.columns
X_pred_sm = disco_party_csv.loc[:,['danceability','acousticness','energy','speechiness','liveness']
y_resp_sm = pd.get_dummies(disco_party_csv.Direction).iloc[:, 1] # dummy encoding for up/down

glm_fit_sm = sm.Logit(y_resp_sm, sm.add_constant(X_pred_sm)).fit()
glm_fit_sm.summary()
#given the inclusion of all the other predictors,
#the only predictor that appears to be statistically
#significant is Lag2 with a P-value of 0.03. 
#all the rest are well above the 5% cutoff of
#significance 

glm_fit_sm.predict()
glm_fit_sm.pred_table()










# load data
sns.pairplot(smarket, hue='Direction')
smarket.corr()

plt.plot(smarket.Volume)



### Figure 4.1 - Default data set
fig = plt.figure(figsize=(12,5))
gs = mpl.gridspec.GridSpec(1, 4)
ax1 = plt.subplot(gs[0,:-2])
ax2 = plt.subplot(gs[0,-2])
ax3 = plt.subplot(gs[0,-1])

# Take a fraction of the samples where target value (default) is 'no'
df_no = df[df.default2 == 0].sample(frac=0.15)
# Take all samples  where target value is 'yes'
df_yes = df[df.default2 == 1]
#df_ = combination of all 'yes' and 15% of 'no'
df_ = df_no.append(df_yes)

ax1.scatter(df_[df_.default == 'Yes'].balance, df_[df_.default == 'Yes'].income, s=40, c='orange', marker='+',
            linewidths=1)
ax1.scatter(df_[df_.default == 'No'].balance, df_[df_.default == 'No'].income, s=40, marker='o', linewidths='1',
            edgecolors='lightblue', facecolors='white', alpha=.6)

ax1.set_ylim(ymin=0)
ax1.set_ylabel('Income')
ax1.set_xlim(xmin=-100)
ax1.set_xlabel('Balance')

c_palette = {'No':'lightblue', 'Yes':'orange'}
sns.boxplot('default', 'balance', data=df, orient='v', ax=ax2, palette=c_palette)
sns.boxplot('default', 'income', data=df, orient='v', ax=ax3, palette=c_palette)
gs.tight_layout(plt.gcf())



    
    
#return a list of my playlists
#playlists = spotify.user_playlists(my_user_id)
#for track in result['tracks']['items']:
    # print(result['tracks']['items'][track]['name'])
    
#need to find the playlist id for the disco party playlist
#print the corresponding playlist id and save it as a variable
#disco_party_id = ''
#for i in range(0, len(playlists['items'])):
    # if playlists['items'][i]['name'] == "disco party":
    #     disco_party_id = playlists['items'][i]['id']
    #     print(playlists['items'][i]['id']) #0l84JP9zEu8NLIGVjHyR1W
 
#spotify has a limit of how many tracks you can pull 
#they limit 100 tracks per playlist
#so maybe i'll go another route
#export track names to a csv and then analyze the 
#tracks from that csv?
#i use a service called Soundiiz to achieve parity
#across all my accounts
#i just easily exported the list of songs from playlist to a csv




























 
 


 
 
    
    
    
    
    
#https://medium.com/gabriel-luz/an%C3%A1lise-de-dados-de-uma-playlist-do-spotify-com-python-e-power-bi-bc848aa0880c
import spotipy
import spotipy.util as util
from spotipy.oauth2 import SpotifyClientCredentials

client_id = "52d3aec17a9044f0a3b0aac1d83eb356"
client_secret = "36e79b71874a437589d269c445815bd3"
redirect_uri = 'https://developer.spotify.com/dashboard/applications/52d3aec17a9044f0a3b0aac1d83eb356'

username = '1218037846'
playlist = '0l84JP9zEu8NLIGVjHyR1W'
client_credentials_manager = SpotifyClientCredentials(client_id = client_id, client_secret = client_secret)
scope = 'user-library-read playlist-read-private'
try:
     token = util.prompt_for_user_token(username, scope, client_id = client_id, client_secret = client_secret, redirect_uri = redirect_uri)
     sp = spotipy.Spotify(auth = token)
except:
    print ('Token is not accessible for' + username)



def get_playlist_audio_features(username, playlist_id, sp):
    offset = 0
    songs = []
    items = []
    ids = []
    while True:
        content = sp.user_playlist_tracks(username, playlist_id, fields=None, limit=100, offset=offset, market=None)
        songs += content['items']
        if content['next'] is not None:
            offset += 100
        else:
            break

    for i in songs:
        ids.append(i['track']['id'])

    index = 0
    audio_features = []
    while index < len(ids):
        audio_features += sp.audio_features(ids[index:index + 50])
        index += 50

    features_list = []
    for features in audio_features:
        features_list.append([features['energy'], features['liveness'],
                              features['tempo'], features['speechiness'],
                              features['acousticness'], features['instrumentalness'],
                              features['time_signature'], features['danceability'],
                              features['key'], features['duration_ms'],
                              features['loudness'], features['valence'],
                              features['mode'], features['type'],
                              features['uri']])

    df = pd.DataFrame(features_list, columns=['energy', 'liveness',
                                              'tempo', 'speechiness',
                                              'acousticness', 'instrumentalness',
                                              'time_signature', 'danceability',
                                              'key', 'duration_ms', 'loudness',
                                              'valence', 'mode', 'type', 'uri'])
    df.to_csv('{}-{}.csv'.format(username, playlist_id), index=False)


