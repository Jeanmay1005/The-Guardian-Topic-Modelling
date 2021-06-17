#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import requests
import json


# In[3]:


apikey = "ad88ad93-b2db-485a-a0c2-82da759612b4" 
baseURL = "https://content.guardianapis.com/search?"
section = "science" 
page = 1

#page through all results using for loop
url_list = []
for page in range(1,566):
    url = baseURL+'section='+section+"&page="+str(page)+"&api-key="+apikey
    url_list += [url]


# In[4]:


def getArticleData(url):
    response = requests.get(url)
    data = json.loads(response.content)
    result = data['response']['results']
    return result


# In[5]:


result = [] #list that contains all results
for i in url_list:
    result = result+[getArticleData(i)]  


# In[7]:


title = []
date = []
for k in result:
    for r in k:
        date += [r['webPublicationDate']]
        title += [r['webTitle']]  


# In[8]:


science_df = pd.DataFrame({'Date':date,'Title':title})
science_df


# In[14]:


import warnings
warnings.filterwarnings("ignore")

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim.utils import tokenize
from gensim.utils import simple_preprocess
from gensim.corpora.textcorpus import remove_stopwords
#from gensim.summarization import keywords
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models


# In[15]:


tokens = list(tokenize(science_df['Title'][0], lowercase = True))


# In[16]:


science_df['terms'] = [remove_stopwords(simple_preprocess(title)) for title in science_df['Title']]


# In[17]:


science_df


# In[18]:


vocab = Dictionary(science_df['terms'])
print(vocab.token2id) # this function dorectly goves out the frequency of the vocabs


# __Use TF-IDF Model on Titles and Find Most Relevant terms__

# In[19]:


# convert corpus to BoW format
corpus = [vocab.doc2bow(terms) for terms in science_df['terms']]  
model = TfidfModel(corpus)# fit a tf-idf model to the corpus
tfidf_doc = model[corpus] # apply model to the first corpus


# In[20]:


def get_tfidf (index):
    term_values = [(vocab[e[0]],e[1]) for e in model[corpus[index]] if e[1]>0]
    srt =  sorted(term_values, key=lambda x: x[1],reverse=True)
    return list(map(lambda x: x[0],srt[:5]))


# __LDA Model of our Corpus__

# In[21]:


# create LDA model witg corpus and vocab, and define the topic numbers
lda_model = LdaModel(corpus = corpus, id2word = vocab, num_topics = 20)


# In[22]:


# show_topic function returns a list with format of [topic number, topic content]
for topic in lda_model.show_topics(num_topics = 3, num_words = 15):
    print("Topic "+str(topic[0])+"\n"+topic[1]+"\n")


# __Get the probability that a certain document belongs to a certain topic__

# In[23]:


doc = science_df['Title'][1]
print("doc:\n",doc)
doc_topics = lda_model.get_document_topics(corpus[2] ,minimum_probability=0.3)
print("doc_topics:\n",doc_topics)

for topic in doc_topics:
    terms = [term for term, prob in lda_model.show_topic(topic[0])]
    print(terms)


# __Visualize the Spread of Topics__

# In[24]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, vocab)
vis


# __Overall Topic Analysis:__
# 
# 
# - One of the most distinctive topic is Topic 1, which contains keywords such as DNA, baby, gender-related terms, medicine, humans. Thus we can understand that this topic may be highly relevant to study in the biological field.
# 
# 
# - The second distinctive topic (Topic No.2) contains terms like nasa, mars, mission, space, life...etc. Which indicates that this topic is quite releveant to explorations into outerspace, especially Mars and moon (whci are the terms, too). 
# 
# 
# - Some topcs are highly overlapped with each other. Topic 5, 7, 8, 19, 20 gather a cluster. And from the terms thay include, we can observe that their shared traits lies in more general scientific terms such as scientists, science, experiements... etc. So their value are rather low  when distincting topics from articles.
# 
# 
# - Overall, in most topics recognized by the model, though we can connect some terms with human knowledge, other terms are compiled by patterns that are out of common senses. This is both a benefit and downside. This means that after calculating the corpus, the machine has found patterns that is difficult for human to find. By utilizing this features, we can efficeintly classify different articles. However, sice we don't know how the model actually differs them, it will be hard for us human to reveal when the algorithm is making a mistake.
