#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 19:35:34 2019

@author: davidmadsen
"""

from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC #, SVC
from statistics import mean, stdev

def Vectorize(vectorizer,
              train,
              test):
    
    train_vec = vectorizer.fit_transform(train)
    
    print(train_vec.shape)
    print(train_vec[0].toarray())
    
    print(len(vectorizer.vocabulary_))
    print(list(vectorizer.vocabulary_.items())[:10])
    
    if (type(test) != type(None)):
        test_vec = vectorizer.transform(test)
    else:
        test_vec = None
    
    return train_vec, test_vec

def SimpleFit(clf,
              X,
              y):
    clf.fit(X, y)
    return clf

def PrintTopNFeatures(N,
                      clf,
                      label_index,
                      label_friendly_name,
                      feature_names,
                      order = -1,
                      log_prob = False):
    print()
    if (log_prob):
        feature_ranks = sorted(zip(10**clf.coef_[label_index], feature_names))
    else:
        feature_ranks = sorted(zip(clf.coef_[label_index], feature_names))
    
    print(label_friendly_name)
    
    if order == -1:
        top_N_features = feature_ranks[(-1 * N):]
    else:
        top_N_features = feature_ranks[:N]
    
#    for f in top_N_features:
#        print(f)
    
    features = []
    prob = []
    
    for f in top_N_features:
        features.append(f[1])
        prob.append(f[0])
    
#    print(features)
#    print()
    
    y_pos = np.arange(len(features))
    plt.barh(y_pos, prob, align='center', alpha=0.5)
    plt.yticks(y_pos, features)
    plt.xticks(rotation=60)
    plt.show()
    
    
def PrintAccuracyReport(clf,
                        X_vec,
                        y,
                        labels):
    print()
    #target_names = [str(l) for l in labels]
    y_pred = clf.predict(X_vec)
    cm = confusion_matrix(y, y_pred)
    print(cm)
    print()
    print(classification_report(y, y_pred))
    print()
    
    
def CrossValidate(vectorizer,
                  classifier,
                  X,
                  y,
                  k):
    pipe = Pipeline([('vect', vectorizer), ('model', classifier)])
    cv = StratifiedKFold(n_splits = k)
    
    pipe.fit(X,y)
    
    return cross_validate(pipe,
                          X,
                          y,
                          cv = cv,
                          scoring = ['accuracy',
                                     'precision_weighted',
                                     'recall_weighted',
                                     'f1_weighted'],
                         return_estimator = True,
                         n_jobs=-1), pipe
                          
def GridSearch(vectorizer,
               classifier,
               X,
               y,
               k,
               param_name):
    print(param_name)
    pipe = Pipeline([('vect', vectorizer), ('model', classifier)])
    print(pipe)
    param_grid = [{
            param_name:[0.01, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 10]}]
    
    print(param_grid)
    
    grid = GridSearchCV(pipe, cv=k, n_jobs = -1, param_grid=param_grid,
                        iid = False)
    return(grid.fit(X, y))
    
def PrintScores(score_key,
                cv_results):
    print()
    print(score_key,
      min(cv_results[score_key]),
      mean(cv_results[score_key]),
      max(cv_results[score_key]))
    print()
    
def PrintCvScores(cv_results):
    PrintScores('test_accuracy', cv_results)
    PrintScores('test_precision_weighted', cv_results)
    PrintScores('test_recall_weighted', cv_results)
    PrintScores('test_f1_weighted', cv_results)
    
# Load the tweets
tweets = pd.read_csv('../data/tweets_labeled.tsv',
                     sep='\t',
                     parse_dates = ['created_at'])

# set up a central time object for localizing game times for Houston, Texas
central_time = pytz.timezone('America/Chicago')

# set the game start of the previous game
# tweets created before this time will be training data.
pre_game_time = datetime(2019, 11, 21, 19, 20, 0, 0, central_time)

# set the start of the game to be tested.
# tweets labeled between 'pre_game_time' and 'game_time'
# will be considered our test data
game_time = datetime(2019, 12, 1, 19, 20, 0, 0, central_time)

# split out the pregame tweets
pregame = tweets[tweets.created_at < pre_game_time]
pregame = pregame[pregame.label != '-']

# split out the game week tweets
game_week = tweets[tweets.created_at >= pre_game_time ]
game_week = game_week[game_week.created_at < game_time]

# get lists of the tweet texts for vectorization
game_week_text = game_week.text.values.astype('U').tolist()
pregame_text = pregame.text.values.astype('U').tolist()

unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
                                           encoding = 'latin-1',
                                           binary = False,
                                           min_df = 1,
                                           stop_words = 'english')

train_vec, test_vec = Vectorize(unigram_count_vectorizer,
                                pregame_text,
                                game_week_text)

# get the labels
pregame_labels = pregame['label'].values.tolist()
game_week_labels = game_week['label'].values.tolist()

# scale the labels back to just 2
pregame_labels_small = []
for l in pregame_labels:
    if l == 'uu':
        pregame_labels_small.append('u')
    elif l == 'dd':
        pregame_labels_small.append('d')
    else:
        pregame_labels_small.append(l)
        
game_week_labels_small = []
for l in game_week_labels:
    if l == 'uu':
        game_week_labels_small.append('u')
    elif l == 'dd':
        game_week_labels_small.append('d')
    else:
        game_week_labels_small.append(l)
        
clf = MultinomialNB()

clf = SimpleFit(clf,
                train_vec,
                pregame_labels_small)

print ("Multinomial NB, Up week, simple labels")
PrintAccuracyReport(clf,
                    test_vec,
                    game_week_labels_small,
                    ['dd','d','u','uu'])

PrintTopNFeatures(10,
                  clf,
                  0,
                  'u',
                  unigram_count_vectorizer.get_feature_names(),
                  -1,
                  True)

PrintTopNFeatures(10,
                  clf,
                  0,
                  'u',
                  unigram_count_vectorizer.get_feature_names(),
                  1,
                  True)

print ("------------------------")
print ()
#PrintTopNFeatures(10,
#                  clf,
#                  0,
#                  'd',
#                  unigram_count_vectorizer.get_feature_names(),
#                  -1,
#                  False)

clf = LinearSVC()

clf = SimpleFit(clf,
                train_vec,
                pregame_labels_small)

print("LinearSVC, upweek, simple labels")

PrintAccuracyReport(clf,
                    test_vec,
                    game_week_labels_small,
                    ['dd','d','u','uu'])

PrintTopNFeatures(10,
                  clf,
                  0,
                  'u',
                  unigram_count_vectorizer.get_feature_names(),
                  -1,
                  False)


PrintTopNFeatures(10,
                  clf,
                  0,
                  'd',
                  unigram_count_vectorizer.get_feature_names(),
                  1,
                  False)

print ("------------------------")
print ()

#PrintTopNFeatures(10,
#                  clf,
#                  0,
#                  'd',
#                  unigram_count_vectorizer.get_feature_names(),
#                  -1,
#                  False)
unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
                                           encoding = 'latin-1',
                                           binary = False,
                                           min_df = 1,
                                           stop_words = 'english')

print ("Performing grid search for optimal C value for LinearSVC")

clf = GridSearch(unigram_count_vectorizer,
                 LinearSVC(),
                 pregame_text,
                 pregame_labels_small,
                 10,
                 'model__C')

best_param = clf.best_params_['model__C']
print("C = ", best_param)
print("------------------------")
print()

print("Performing 10-fold CV on a up week using 'best C'")
unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
                                           encoding = 'latin-1',
                                           binary = False,
                                           min_df = 1,
                                           stop_words = 'english')

clf, pipe = CrossValidate(unigram_count_vectorizer,
                           LinearSVC(C=best_param),
                           pregame_text,
                           pregame_labels_small,
                           10)
PrintCvScores(clf)

best_estimator = clf['estimator'][np.argmax(clf['test_f1_weighted'])]

PrintAccuracyReport(best_estimator._final_estimator,
                    best_estimator.steps[0][1].transform(game_week_text),
                    game_week_labels_small,
                    ['dd','d','u','uu'])


#print("Performing 10-fold CV on a up week using C=1")
#
#unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
#                                           encoding = 'latin-1',
#                                           binary = False,
#                                           min_df = 1,
#                                           stop_words = 'english')
#
#clf, pipe = CrossValidate(unigram_count_vectorizer,
#                           LinearSVC(C=1),
#                           pregame_text,
#                           pregame_labels_small,
#                           10)
#PrintCvScores(clf)
#
#clf = clf['estimator'][np.argmax(clf['test_f1_weighted'])]._final_estimator
#PrintAccuracyReport(clf,
#                    test_vec,
#                    game_week_labels_small,
#                    ['dd','d','u','uu'])


pre_game_time = datetime(2019, 11, 3, 8, 30, 0, 0, central_time)
game_time = datetime(2019, 11, 17, 12, 0, 0, 0, central_time)

pregame = tweets[tweets.created_at < pre_game_time]
pregame = pregame[pregame.label != '-']
game_week = tweets[tweets.created_at >= pre_game_time ]
game_week = game_week[game_week.created_at < game_time]

game_week_text = game_week.text.values.astype('U').tolist()
pregame_text = pregame.text.values.astype('U').tolist()

pregame_labels = pregame['label'].values.tolist()
game_week_labels = game_week['label'].values.tolist()

pregame_labels_small = []
for l in pregame_labels:
    if l == 'uu':
        pregame_labels_small.append('u')
    elif l == 'dd':
        pregame_labels_small.append('d')
    else:
        pregame_labels_small.append(l)
        
game_week_labels_small = []
for l in game_week_labels:
    if l == 'uu':
        game_week_labels_small.append('u')
    elif l == 'dd':
        game_week_labels_small.append('d')
    else:
        game_week_labels_small.append(l)
        
        
unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
                                           encoding = 'latin-1',
                                           binary = False,
                                           min_df = 1,
                                           stop_words = 'english')

print ("Performing grid search for optimal C value for LinearSVC")

clf = GridSearch(unigram_count_vectorizer,
                 LinearSVC(),
                 pregame_text,
                 pregame_labels_small,
                 10,
                 'model__C')

best_param = clf.best_params_['model__C']
print("C = ", best_param)
print("------------------------")
print()

print("Performing 10-fold CV on a down week using 'best C'")
unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
                                           encoding = 'latin-1',
                                           binary = False,
                                           min_df = 1,
                                           stop_words = 'english')

clf, pipe = CrossValidate(unigram_count_vectorizer,
                           LinearSVC(C=best_param),
                           pregame_text,
                           pregame_labels_small,
                           10)
PrintCvScores(clf)
best_estimator = clf['estimator'][np.argmax(clf['test_f1_weighted'])]

PrintAccuracyReport(best_estimator._final_estimator,
                    best_estimator.steps[0][1].transform(game_week_text),
                    game_week_labels_small,
                    ['dd','d','u','uu'])

#print("Performing 10-fold CV on a down week using C=1")
#
#unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
#                                           encoding = 'latin-1',
#                                           binary = False,
#                                           min_df = 1,
#                                           stop_words = 'english')
#
#clf, pipe = CrossValidate(unigram_count_vectorizer,
#                           LinearSVC(C=1),
#                           pregame_text,
#                           pregame_labels_small,
#                           10)
#PrintCvScores(clf)
#best_estimator = clf['estimator'][np.argmax(clf['test_f1_weighted'])]
#
#PrintAccuracyReport(best_estimator._final_estimator,
#                    best_estimator._transform(game_week_text),
#                    game_week_labels_small,
#                    ['dd','d','u','uu'])


unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
                                           encoding = 'latin-1',
                                           binary = False,
                                           min_df = 1,
                                           stop_words = 'english')

train_vec, test_vec = Vectorize(unigram_count_vectorizer,
                                pregame_text,
                                game_week_text)


clf = MultinomialNB()

clf = SimpleFit(clf,
                train_vec,
                pregame_labels_small)

print("Multinomial NB, down week, simple labels")

PrintAccuracyReport(clf,
                    test_vec,
                    game_week_labels_small,
                    ['dd','d','u','uu'])

PrintTopNFeatures(10,
                  clf,
                  0,
                  'd',
                  unigram_count_vectorizer.get_feature_names(),
                  -1,
                  True)

PrintTopNFeatures(10,
                  clf,
                  0,
                  'd',
                  unigram_count_vectorizer.get_feature_names(),
                  1,
                  True)

print ("------------------------")
print ()

#PrintTopNFeatures(10,
#                  clf,
#                  0,
#                  'd',
#                  unigram_count_vectorizer.get_feature_names(),
#                  -1,
#                  False)

clf = LinearSVC()

clf = SimpleFit(clf,
                train_vec,
                pregame_labels_small)

print ("LinearSVC, down week, simple labels")

PrintAccuracyReport(clf,
                    test_vec,
                    game_week_labels_small,
                    ['dd','d','u','uu'])

PrintTopNFeatures(10,
                  clf,
                  0,
                  'd',
                  unigram_count_vectorizer.get_feature_names(),
                  -1,
                  False)

PrintTopNFeatures(10,
                  clf,
                  0,
                  'd',
                  unigram_count_vectorizer.get_feature_names(),
                  1,
                  False)

print ("------------------------")
print ()

nfl_sent_pd = pd.read_csv('../data/nfl_sent_labeled.csv', parse_dates=['created_at'])
#nfl_sent_pd['created_at'] = pd.to_datetime(nfl_sent_pd['created_at'],
#             format = '%a %b %d %H:%M:%S %z %Y')

sent_pregame = []
sent_gametime = []
sent_pregame_labels =[]
sent_gametime_labels = []

for row in nfl_sent_pd.itertuples():
    if row[2] < pre_game_time:
        sent_pregame.append((row[3], row[4], row[5]))
        sent_pregame_labels.append(row[6])
    elif row[2] >= pre_game_time and row[2] < game_time:
        sent_gametime.append((row[3], row[4], row[5]))
        sent_gametime_labels.append(row[6])
        
sent_pregame_labels_small = []
for l in sent_pregame_labels:
    if l == 'uu':
        sent_pregame_labels_small.append('u')
    elif l == 'dd':
        sent_pregame_labels_small.append('d')
    else:
        sent_pregame_labels_small.append(l)
        
sent_game_week_labels_small = []
for l in sent_gametime_labels:
    if l == 'uu':
        sent_game_week_labels_small.append('u')
    elif l == 'dd':
        sent_game_week_labels_small.append('d')
    else:
        sent_game_week_labels_small.append(l)
        
clf = SimpleFit(LinearSVC(),
                sent_pregame,
                sent_pregame_labels_small)

print("LinearSVC sentiment down week")

PrintAccuracyReport(clf,
                    sent_gametime,
                    sent_game_week_labels_small,
                    ['d', 'u'])

print ("------------------------")
print ()


pre_game_time = datetime(2019, 11, 21, 19, 20, 0, 0, central_time)
game_time = datetime(2019, 12, 1, 19, 20, 0, 0, central_time)

sent_pregame = []
sent_gametime = []
sent_pregame_labels =[]
sent_gametime_labels = []

for row in nfl_sent_pd.itertuples():
    if row[2] < pre_game_time:
        sent_pregame.append((row[3], row[4], row[5]))
        sent_pregame_labels.append(row[6])
    elif row[2] >= pre_game_time and row[2] < game_time:
        sent_gametime.append((row[3], row[4], row[5]))
        sent_gametime_labels.append(row[6])
     
sent_pregame_labels_small = []
for l in sent_pregame_labels:
    if l == 'uu':
        sent_pregame_labels_small.append('u')
    elif l == 'dd':
        sent_pregame_labels_small.append('d')
    else:
        sent_pregame_labels_small.append(l)
        
sent_game_week_labels_small = []
for l in sent_gametime_labels:
    if l == 'uu':
        sent_game_week_labels_small.append('u')
    elif l == 'dd':
        sent_game_week_labels_small.append('d')
    else:
        sent_game_week_labels_small.append(l)
              
clf = SimpleFit(LinearSVC(),
                sent_pregame,
                sent_pregame_labels_small)

print("LinearSVC sentiment up week")

PrintAccuracyReport(clf,
                    sent_gametime,
                    sent_game_week_labels_small,
                    ['d', 'u'])

print ("------------------------")
print ()

tweets = pd.read_csv('../data/tweets_points.tsv',
                     sep='\t',
                     parse_dates = ['created_at'])

points = pd.read_csv('../data/Deshaun_Watson_Scoring.csv',
                     parse_dates = ['Date'])

points['Date'] = points['Date'].dt.tz_localize(central_time)

reg_scores = []
reg_points = []
reg_models = []

for i in range(0, len(points.index) - 1):
    pre_game_time = points.iloc[i]['Date']
    game_time = points.iloc[i+1]['Date']
    print ()
    print ('Linear regression for', game_time)
    

    pregame = tweets[tweets.created_at < pre_game_time]
    pregame = pregame[pregame.label != '-']
    game_week = tweets[tweets.created_at >= pre_game_time ]
    game_week = game_week[game_week.created_at < game_time]

    game_week_text = game_week.text.values.astype('U').tolist()
    pregame_text = pregame.text.values.astype('U').tolist()

    unigram_count_vectorizer = TfidfVectorizer(use_idf=False,
                                               encoding = 'latin-1',
                                               binary = False,
                                               min_df = 1,
                                               stop_words = 'english',
                                               ngram_range = (1,1))

    train_vec, test_vec = Vectorize(unigram_count_vectorizer,
                                    pregame_text,
                                    game_week_text)

    pregame_labels = pregame['label'].values.tolist()
    game_week_labels = game_week['label'].values.tolist()

    #train_arr = train_vec.toarray()
    #test_arr = test_vec.toarray()

    reg = LinearRegression()

    reg = SimpleFit(reg,
                    train_vec,
                    pregame_labels)

    reg_scores.append((game_time,
                       reg.score(train_vec,
                                 pregame_labels)))
    
    reg_predict = reg.predict(test_vec)
    
    reg_points.append((game_time,
                       points.iloc[i+1]['FPTS'],
                       mean(reg_predict),
                       stdev(reg_predict)))
    
    reg_models.append((game_time, reg))

print(reg_scores)
print(reg_points)
print()
    
nfl_sent_pd = pd.read_csv('../data/nfl_sent_points_labeled.csv',
                          parse_dates=['created_at'])

sent_reg_points = []
sent_reg_models = []
sent_reg_scores = []

for i in range(0, len(points.index) - 1):
    print (i, i+1)
    pre_game_time = points.iloc[i]['Date']
    game_time = points.iloc[i+1]['Date']
    
    print (pre_game_time, game_time)
    
    sent_pregame = []
    sent_pregame_labels = []
    
    sent_gametime = []
    sent_gametime_labels = []
    print ()
    print ('Sentiment Linear regression for', game_time)
    
    for row in nfl_sent_pd.itertuples():
        if row[2] < pre_game_time:
            sent_pregame.append((row[3], row[4], row[5]))
            sent_pregame_labels.append(row[6])
        elif row[2] >= pre_game_time and row[2] < game_time:
            sent_gametime.append((row[3], row[4], row[5]))
            sent_gametime_labels.append(row[6])
     
    sent_reg = SimpleFit(LinearRegression(),
                         sent_pregame,
                         sent_pregame_labels)
    
    sent_reg_scores.append((game_time,
                            sent_reg.score(sent_pregame,
                                           sent_pregame_labels)))
        
    sent_reg_points.append((game_time,
                       float(points[points.Date == game_time]['FPTS']),
                       mean(sent_reg.predict(sent_gametime)),
                       stdev(sent_reg.predict(sent_gametime))))
    
    sent_reg_models.append((game_time, sent_reg))
    
    
print(sent_reg_scores)
print(sent_reg_points)