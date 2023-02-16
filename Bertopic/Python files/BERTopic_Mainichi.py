#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#0
import numpy as np
import pandas as pd
import pickle

with open(r"Arti_dict_Price.dump", mode='rb') as f:
    Arti_dict=pickle.load(f)


# In[2]:


Arti_dict


# In[6]:


df = pd.DataFrame(Arti_dict)
df


# In[3]:


# ストップワード辞書のダウンロード
import os
import urllib.request

def download_stopwords(path):
    url = 'http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt'
    if os.path.exists(path):
        print('File already exists.')
    else:
        print('Downloading...')
        # Download the file from `url` and save it locally under `file_name`:
        urllib.request.urlretrieve(url, path)


# In[4]:


# ストップワードの作成
def create_stopwords(file_path):
    stop_words = []
    for w in open(path, "r", encoding="utf-8"):
        w = w.replace('\n','')
        if len(w) > 0:
          stop_words.append(w)
    return stop_words    

path = "stop_words.txt"
download_stopwords(path)
stop_words = create_stopwords(path)


# In[25]:


import re
import demoji
import neologdn
import MeCab

wakati = MeCab.Tagger("-Owakati")

def clean_text(x):
  x = str(x)
  x = neologdn.normalize(x)
  x = x.lower()
  x = re.sub(r'#[A-Za-z0-9]*', '', x)
  x = re.sub(r'https*://.*', '', x)
  x = re.sub(r'@[A-Za-z0-9]+', '', x)
  x = demoji.replace(string=x, repl='')
  #x = re.sub(r'[%s]' % re.escape('!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~“…”’'), ' ', x)
  x = re.sub(r'[%s]' % re.escape('!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％◇'), '', x)
  x = re.sub("[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", '', x)
  x = re.sub(r'\d+', '', x)
  x = re.sub(r'\n+', '', x)
  x = re.sub(r'\s{2,}', '', x)
  x = re.sub("sep", '', x)
  x = re.sub("cls", '', x)
  x = wakati.parse(x)
  
  return x


df['clean_text'] = df.Body.apply(clean_text)
df.head()


# In[8]:


df.shape


# In[27]:


bodys = df.clean_text.to_list()
timestamp = df.Time.to_list()


# In[19]:


from transformers import BertJapaneseTokenizer, BertModel
import torch


class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)


# In[20]:


model = SentenceBertJapanese("sonoisa/sentence-bert-base-ja-mean-tokens-v2")


# In[28]:


from bertopic import BERTopic
topic_model = BERTopic(embedding_model=model)
topics, probs = topic_model.fit_transform(bodys)


# In[ ]:


f = open('topics.txt', 'wb')
pickle.dump(topics, f)


# In[13]:


f = open("./topics.txt","rb")
topics = pickle.load(f)

f = open("./probs.txt","rb")
probs = pickle.load(f)


# In[14]:


f = open('probs.txt', 'wb')
pickle.dump(probs, f)


# In[37]:


f = open('topic_model.txt', 'wb')
pickle.dump(topic_model, f)


# In[38]:


f = open("./topic_model.txt","rb")
topic_model = pickle.load(f)


# In[39]:


topic_model.get_topic_info()


# In[30]:


topic_model.visualize_barchart()


# In[33]:


timestamp


# In[35]:


topics_over_time = topic_model.topics_over_time(bodys, topics, timestamp, nr_bins=20)

