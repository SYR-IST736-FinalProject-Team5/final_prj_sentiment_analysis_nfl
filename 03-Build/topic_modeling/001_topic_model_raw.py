#!/usr/bin/env python
# coding: utf-8

# In[2]:


#imports
import os

import pandas as pd
import numpy as np

import pyLDAvis.sklearn as LDAvis
import pyLDAvis

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.decomposition import NMF, LatentDirichletAllocation

import warnings
warnings.filterwarnings('ignore')


# In[3]:


# functions
def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])


# In[4]:


#file_path = "C:\\Users\\dvjr2\\Google Drive\\Documents\\Syracuse\\IST_736_TextMining\\Project\\tables"
file_path = "C:\\Users\\dvjr2\\Google Drive\\Documents\\Syracuse\\IST_736_TextMining\\Project\\raw"

file_list = []
essay_list = []
for item in os.listdir(file_path):
    
    file_list.append(file_path + "\\" + item)  
    essay_list.append(item)
    
print(file_list[0])
print(essay_list[0])


# In[5]:


no_features = 500

#lda_cv = CountVectorizer(input = 'filename', analyzer = 'word', stop_words = 'english', lowercase = True, 
#encoding="ISO-8859-1")
lda_cv = CountVectorizer(input = 'filename', analyzer = 'word', stop_words = 'english', lowercase = True, 
                         max_df = 0.95, min_df = 2, max_features = no_features, encoding="ISO-8859-1")
#lda_cv = CountVectorizer(input = 'filename', analyzer = 'word', lowercase = True, encoding="ISO-8859-1")
#lda_cv = CountVectorizer(input = 'filename', encoding="ISO-8859-1")


# In[6]:


CV = lda_cv.fit_transform(file_list)
lda_columns = lda_cv.get_feature_names()
df_corpus_lda = pd.DataFrame(CV.toarray(),columns = lda_columns)
df_corpus_lda.head()


# In[7]:


no_topics = 5
max_iterations = 10
learn_off = 50
random = 0

lda_model = LatentDirichletAllocation(n_components = no_topics, max_iter = max_iterations, learning_method='online',
                                         learning_offset = learn_off,random_state = random)
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model.fit_transform(CV)


# In[8]:


print(LDA_DH_Model.shape) # docs | topics
print(LDA_DH_Model[0])


# In[9]:


print_topics(lda_model, lda_cv)


# In[10]:


panel = LDAvis.prepare(lda_model, CV, lda_cv, mds='tsne')
'''
Note: if you're in the IPython notebook, pyLDAvis.show() is not the best command
      to use. Consider using pyLDAvis.display(), or pyLDAvis.enable_notebook().
'''
pyLDAvis.show(panel)


# In[ ]:





# In[ ]:




