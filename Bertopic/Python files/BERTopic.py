#!/usr/bin/env python
# coding: utf-8

# # **BERTopic - Tutorial**
# We start with installing bertopic from pypi before preparing the data. 
# 
# **NOTE**: Make sure to select a GPU runtime. Otherwise, the model can take quite some time to create the document embeddings!

# # **Prepare data**
# For this example, we use the popular 20 Newsgroups dataset which contains roughly 18000 newsgroups posts on 20 topics.

# In[1]:


from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']


# In[3]:


print(type(docs))


# In[7]:


docs[3]


# In[9]:


len(docs)


# # **Create Topics**
# We select the "english" as the main language for our documents. If you want a multilingual model that supports 50+ languages, please select "multilingual" instead. 

# In[8]:


model = BERTopic(language="english")
topics, probs = model.fit_transform(docs)


# In[13]:


len(probs)


# In[35]:


model.get_topic_freq().head(5)


# -1 refers to all outliers and should typically be ignored. Next, let's take a look at the most frequent topic that was generated:

# In[34]:


model.get_topic(0)[:10]


# Note that the model is stocastich which mmeans that the topics might differ across runs. 
# 
# For a full list of support languages, see the values below:

# In[8]:


from bertopic.backend import languages
print(languages)


# ## Attributes

# There are a number of attributes that you can access after having trained your BERTopic model:
# 
# 
# | Attribute | Description |
# |------------------------|---------------------------------------------------------------------------------------------|
# | topics_               | The topics that are generated for each document after training or updating the topic model. |
# | probabilities_ | The probabilities that are generated for each document if HDBSCAN is used. |
# | topic_sizes_           | The size of each topic                                                                      |
# | topic_mapper_          | A class for tracking topics and their mappings anytime they are merged/reduced.             |
# | topic_representations_ | The top *n* terms per topic and their respective c-TF-IDF values.                             |
# | c_tf_idf_              | The topic-term matrix as calculated through c-TF-IDF.                                       |
# | topic_labels_          | The default labels for each topic.                                                          |
# | custom_labels_         | Custom labels for each topic as generated through `.set_topic_labels`.                                                               |
# | topic_embeddings_      | The embeddings for each topic if `embedding_model` was used.                                                              |
# | representative_docs_   | The representative documents for each topic if HDBSCAN is used.                                                |
# 
# For example, to access the predicted topics for the first 10 documents, we simply run the following:

# In[19]:


model.topics_[:10]


# # **Embedding model**
# You can select any model from `sentence-transformers` and use it instead of the preselected models by simply passing the model through  
# BERTopic with `embedding_model`:

# In[ ]:


# st_model = BERTopic(embedding_model="xlm-r-bert-base-nli-stsb-mean-tokens")


# Click [here](https://www.sbert.net/docs/pretrained_models.html) for a list of supported sentence transformers models.  
# 

# # **Visualize Topics**
# After having trained our `BERTopic` model, we can iteratively go through perhaps a hundred topic to get a good 
# understanding of the topics that were extract. However, that takes quite some time and lacks a global representation. 
# Instead, we can visualize the topics that were generated in a way very similar to 
# [LDAvis](https://github.com/cpsievert/LDAvis):

# In[1]:


model.visualize_topics()


# # **Topic Reduction**
# Finally, we can also reduce the number of topics after having trained a BERTopic model. The advantage of doing so, 
# is that you can decide the number of topics after knowing how many are actually created. It is difficult to 
# predict before training your model how many topics that are in your documents and how many will be extracted. 
# Instead, we can decide afterwards how many topics seems realistic:
# 
# 
# 
# 

# In[11]:


model.reduce_topics(docs, nr_topics=60)


# # **Topic Representation**
# When you have trained a model and viewed the topics and the words that represent them,
# you might not be satisfied with the representation. Perhaps you forgot to remove
# stop_words or you want to try out a different n_gram_range. We can use the function `update_topics` to update 
# the topic representation with new parameters for `c-TF-IDF`: 
# 

# In[13]:


model.update_topics(docs, n_gram_range=(1, 3))


# # **Search Topics**
# After having trained our model, we can use `find_topics` to search for topics that are similar 
# to an input search_term. Here, we are going to be searching for topics that closely relate the 
# search term "gpu". Then, we extract the most similar topic and check the results: 

# In[24]:


similar_topics, similarity = model.find_topics("gpu", top_n=5); similar_topics


# In[25]:


model.get_topic(18)


# # **Model serialization**
# The model and its internal settings can easily be saved. Note that the documents and embeddings will not be saved. However, UMAP and HDBSCAN will be saved. 

# In[ ]:


# Save model
model.save("my_model")


# In[ ]:


# Load model
my_model = BERTopic.load("my_model")

