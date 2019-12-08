#!/usr/bin/env python
# coding: utf-8

# In[12]:


import nltk
import pandas as pd
import sklearn
import re  
import os

import pyLDAvis.sklearn as LDAvis
import pyLDAvis

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
## For Stemming


from sklearn.decomposition import NMF, LatentDirichletAllocation

import warnings
warnings.filterwarnings('ignore')


# In[13]:


# functions
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])


# In[14]:


#csv_file = 'tables_v2/v2_coach_search_result_tweet_text_data_unique.csv'
#csv_file = 'tables_v2/v2_nfl_tweets_master.csv'
#csv_file = 'tables_v2/v2_player_search_result_tweet_text_data_unique.csv'
#csv_file = 'tables_v2/v2_team_search_result_tweet_text_data_unique.csv'


#csv_file = 'tables/coach_search_result_tweet_text_data_unique.csv'
csv_file = 'tables/nfl_tweets_master.csv'
#csv_file = 'tables/player_search_result_tweet_text_data_unique.csv'
#csv_file = 'tables/team_search_result_tweet_text_data_unique.csv'

df_coaches = pd.read_csv(csv_file)


# In[15]:


df_coaches.head()


# In[16]:


my_list=[] # list to store results
shortword = re.compile(r'\W*\b\w{1,1}\b') # filtering out all words with len <= 1
my_list = [shortword.sub('', df_coaches.iloc[i,0]) for i in range(len(df_coaches))]


# In[17]:


no_features = 1000
lda_cv = CountVectorizer(input="content", analyzer = 'word', stop_words = 'english', lowercase = True, 
                         max_df = 0.95, min_df = 2, max_features = no_features, encoding="ISO-8859-1")
cv = lda_cv.fit_transform(my_list)


# In[18]:


CV = lda_cv.fit_transform(my_list)
lda_columns = lda_cv.get_feature_names()
df_corpus_lda = pd.DataFrame(CV.toarray(),columns = lda_columns)
df_corpus_lda.head()


# In[19]:


no_topics = 5
max_iterations = 10
learn_off = 50
random = 0

lda_model = LatentDirichletAllocation(n_components = no_topics, max_iter = max_iterations, learning_method='online',
                                         learning_offset = learn_off,random_state = random)
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model.fit_transform(CV)


# In[20]:


print(LDA_DH_Model.shape) # docs | topics
print(LDA_DH_Model[0])


# In[21]:


print_topics(lda_model, lda_cv)


# In[22]:


panel = LDAvis.prepare(lda_model, CV, lda_cv, mds='tsne')
'''
Note: if you're in the IPython notebook, pyLDAvis.show() is not the best command
      to use. Consider using pyLDAvis.display(), or pyLDAvis.enable_notebook().
'''
pyLDAvis.show(panel)


# In[ ]:




