import numpy as np
import pandas as pd
import math

import pathlib
import logging
from smart_open import open
import re

import gensim
from gensim import corpora
from collections import defaultdict
from tqdm import tqdm

import pyLDAvis
import pyLDAvis.gensim_models

from wordcloud import WordCloud
import matplotlib.pylab as plt

import MeCab


df = pd.DataFrame(columns=["category", "url", "time", "title", "text"])

for file_path in pathlib.Path("...\\text").glob("**\\*.txt"):
    f_path = pathlib.Path(file_path)
    file_name = f_path.name
    category_name = f_path.parent.name

    if file_name in ["CHANGES.txt", "README.txt", "LICENSE.txt"]:
        continue

    with open(file_path, "r") as f:
        text_all = f.read()
        text_lines = text_all.split("\n")
        url, time, title, *article = text_lines
        article = "\n".join(article)

        df.loc[file_name] = [category_name, url, time, title, article]

df.reset_index(inplace=True)
df.rename(columns={"index": "filename"}, inplace=True)

df.to_csv("...\\text\\livedoor_news_corpus.csv", encoding="utf-8_sig", index=None)

df = pd.read_csv("...\\text\\livedoor_news_corpus.csv")
df["text"] = df["text"].str.normalize("NFKC")
df["text"] = df["text"].str.lower()
kana_re = re.compile("^[ぁ-ゖ]+$")

tagger = MeCab.Tagger("...\\mecab-ipadic-neologd")

token_list = []
for i in range(0, len(df["text"])):
    parsed_lines = tagger.parse(df["text"][i]).split("\n")[:-2]
surfaces = [l.split('\t')[0] for l in parsed_lines]
token_list.append([t for t in surfaces if not kana_re.match(t)])

f = open("...\\Japanese.txt")
stopword = f.read()
f.close()
texts = []
stop_words = set(stopword.split())
for i in range(0, len(df["text"])):
    texts.append([word for word in token_list[i] if word not in stop_words])

frequency = defaultdict(int)

for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 9] for text in texts]

dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=3, no_above=0.8)

corpus = [dictionary.doc2bow(t) for t in texts]
#########変更部分###########################
tfidf = gensim.models.TfidfModel(corpus) ##
corpus_tfidf = tfidf[corpus]             ##
###########################################
start = 2
limit = 22
step = 1

coherence_vals = []
perplexity_vals = []

for n_topic in tqdm(range(start, limit, step)):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=n_topic, alpha='symmetric', random_state=0)
perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus_tfidf)))
coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
coherence_vals.append(coherence_model_lda.get_coherence())

x = range(start, limit, step)

fig, ax1 = plt.subplots(figsize=(12,5))

c1 = 'darkturquoise'
ax1.plot(x, coherence_vals, 'o-', color=c1)
ax1.set_xlabel('Num Topics')
ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)

c2 = 'slategray'
ax2 = ax1.twinx()
ax2.plot(x, perplexity_vals, 'o-', color=c2)
ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)

ax1.set_xticks(x)
fig.tight_layout()

plt.savefig('...\\figure.png')
##############################################################
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus_tfidf, id2word=dictionary, num_topics=9, alpha="symmetric", random_state=0)

fig, axs = plt.subplots(ncols=2, nrows=math.ceil(lda_model.num_topics/2), figsize=(16,20))
axs = axs.flatten()

def color_func(word, font_size, position, orientation, random_state, font_path):
    return 'darkturquoise'

FONT_PATH = "...\\anaconda3\\Lib\\site-packages\\matplotlib\\mpl-data\\fonts\\ttf\\ipaexg.ttf"

for i, t in enumerate(range(lda_model.num_topics)):

    x = dict(lda_model.show_topic(t, 30))
    im = WordCloud(
        font_path=FONT_PATH,
        background_color='black',
        color_func=color_func,
        max_words=4000,
        width=300, height=300,
        random_state=0
    ).generate_from_frequencies(x)
    axs[i].imshow(im.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)
    axs[i].axis('off')
    axs[i].set_title('Topic '+str(t))

# vis
plt.tight_layout()

# save as png
plt.savefig('..\\figure.png')