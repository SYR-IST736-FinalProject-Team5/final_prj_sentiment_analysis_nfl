#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:21:27 2019

@author: davidmadsen
"""
from datetime import timezone
import os
import pandas as pd
import pytz

path = os.path.join ('..','Texans_fantasy_point_labels.csv' )

scores = pd.read_csv(path)
scores['Date'] = pd.to_datetime(scores['Date'], format='%m/%d/%y %I:%M %p')

central_time = pytz.timezone('America/Chicago')
scores['Date'] = scores['Date'].dt.tz_localize(central_time)

path = os.path.join('~',
                    'Downloads',
                    'nfl_tweets_master.csv')

texans_tweets = pd.read_csv(path)

texans_tweets['created_at'] = pd.to_datetime(texans_tweets['created_at'],
             format = '%a %b %d %H:%M:%S %z %Y')

path = os.path.join ('..', 
                     'text_mine_sentiment_players',
                     'player_articles_texans.tsv')

texans_articles = pd.read_csv(path, sep='\t')
texans_articles['published_at'] = pd.to_datetime(
        texans_articles['published_at'],
        format = '%Y-%m-%dT%H:%M:%SZ')

texans_articles['published_at'] = texans_articles['published_at'].dt.tz_localize(
        timezone.utc)

path = os.path.join('..','Deshaun_Watson_Scoring.csv')
points = pd.read_csv(path)

points['Date'] = pd.to_datetime(points['Date'], format='%m/%d/%y %I:%M %p')
points['Date'] = points['Date'].dt.tz_localize(central_time)

texans_articles_labeled = []

score_columns = scores.columns

score_dates = scores['Date'].tolist()

for row in texans_articles.itertuples():
    # some of the player names came in with a stray leading space
    player_name = row[3].lstrip()
    # just skip players for whom we don't have labels
    if player_name not in score_columns:
        continue
    columns_index = score_columns.tolist().index(player_name)
    print(columns_index)
    for score_row in scores.itertuples():
        if score_row[1] > row[7]:
            print(row[7], score_row[1])
            article_dict = {
                    'published_at':row[7],
                    'player_name':player_name,
                    'title':row[4],
                    'description':row[5],
                    'label':score_row[columns_index + 1]}
            texans_articles_labeled.append(article_dict)    
            break
            
texans_pd = pd.DataFrame(texans_articles_labeled)     
texans_pd.to_csv('texans_labled.tsv', sep='\t')

tweet_list = []
tweet_points_list = []

player_name = 'Deshaun Watson'
columns_index = score_columns.tolist().index(player_name)
points_index = points.columns.tolist().index('FPTS')
   
for row in texans_tweets.itertuples():
    for score_row in scores.itertuples():
        print(score_row[1], row[2])
        i = 0
        if score_row[1] > row[2]:
            tweet_dict = {
                    'id':row[1],
                    'created_at':row[2],
                    'date':row[3],
                    'time':row[4],
                    'user':row[5],
                    'text':row[6],
                    'favorite_count':row[7],
                    'year':row[8],
                    'month':row[9],
                    'day_of_month':row[10],
                    'day_of_week':row[11],
                    'label':score_row[columns_index + 1]}
            tweet_list.append(tweet_dict)
            break
            
tweet_pd = pd.DataFrame(tweet_list)
tweet_pd.to_csv('tweets_labeled.tsv', sep='\t')

for row in texans_tweets.itertuples():
    for points_row in points.itertuples():
        print(points_row[1], row[2])
        i = 0
        if points_row[1] > row[2]:
            tweet_dict = {
                    'id':row[1],
                    'created_at':row[2],
                    'date':row[3],
                    'time':row[4],
                    'user':row[5],
                    'text':row[6],
                    'favorite_count':row[7],
                    'year':row[8],
                    'month':row[9],
                    'day_of_month':row[10],
                    'day_of_week':row[11],
                    'label':points_row[points_index + 1]}
            tweet_points_list.append(tweet_dict)
            break
        
tweet_points_pd = pd.DataFrame(tweet_points_list)
tweet_points_pd.to_csv('tweets_points.tsv', sep='\t')

path = os.path.join('~',
                    'Downloads',
                    'nfl_master_sent_merged_timeseries.csv')

nfl_ts = pd.read_csv(path)

nfl_ts['created_at'] = pd.to_datetime(nfl_ts['created_at'],
             format = '%a %b %d %H:%M:%S %z %Y')

nfl_ts_list = []
nfl_ts_points_list = []
for row in nfl_ts.itertuples():
    sent_dict = eval(row[15])
    for score_row in scores.itertuples():
        print(score_row[1], row[2])
        if score_row[1] > row[2]:
            nfl_ts_dict = {
                    'created_at':row[2],
                    'neg':sent_dict['neg'],
                    'neu':sent_dict['neu'],
                    'pos':sent_dict['pos'],
                    'label':score_row[columns_index + 1]
                    }
            nfl_ts_list.append(nfl_ts_dict)
            nfl_ts_points_dict = {
                    'created_at':row[2],
                    'neg':sent_dict['neg'],
                    'neu':sent_dict['neu'],
                    'pos':sent_dict['pos'],
                    'label':float(points[points.Date == score_row[1]]['FPTS'])
                    }            
            nfl_ts_points_list.append(nfl_ts_points_dict)
            break
    
nfl_ts_data = pd.DataFrame(nfl_ts_list)
nfl_ts_data.to_csv('nfl_sent_labeled.csv')

nfl_ts_points_pd = pd.DataFrame(nfl_ts_points_list)
nfl_ts_points_pd.to_csv('nfl_sent_points_labeled.csv')