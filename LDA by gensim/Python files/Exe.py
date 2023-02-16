import pandas as pd
import pathlib

# データフレームを用意
df = pd.DataFrame(columns=["category", "url", "time", "title", "text"])

# ファイルを読み込む
for file_path in pathlib.Path("C:/Users/r1ryo/Downloads/text").glob("**/*.txt"):
    f_path = pathlib.Path(file_path)
    file_name = f_path.name
    category_name = f_path.parent.name

    # 記事に関係ないファイルはスキップ
    if file_name in ["CHANGES.txt", "README.txt", "LICENSE.txt"]:
        continue

# ファイルを読み込み, 改行毎に分離させる
    with open(file_path, "r") as f:
        text_all = f.read()
        text_lines = text_all.split("\n")
        url, time, title, *article = text_lines
        article = "\n".join(article)

    # 分離させたもの毎にデータフレームに書き込み
        df.loc[file_name] = [category_name, url, time, title, article]

# インデックスに使用していたファイル名を列の1つにする。
df.reset_index(inplace=True)
df.rename(columns={"index": "filename"}, inplace=True)

# CSVファイルに保存
df.to_csv("C:/Users/r1ryo/DataspellProjects/Python/LDA-model-by-gensim/Database/data.csv", encoding="utf-8_sig", index=None)
#%%
# 計算系ライブラリ
import numpy as np
import pandas as pd
import math

# データ読み込み
import pathlib
from smart_open import open
import re

# MeCabのインポート
import MeCab

# gensim関連
import gensim
from gensim import corpora
from collections import defaultdict
from tqdm import tqdm

# LDAモデル
import pyLDAvis
import pyLDAvis.gensim_models
import logging

# 表示用ライブラリ
from wordcloud import WordCloud
import matplotlib.pylab as plt
#%%
df = pd.read_csv("C:/Users/r1ryo/DataspellProjects/Python/LDA-model-by-gensim/Database/data.csv")
# ユニコード正規化（テキストをPC内部で扱いやすくするための処理）
df["text"] = df["text"].str.normalize("NFKC")
# アルファベットを小文字に統一
df["text"] = df["text"].str.lower()
# ひらがなのみの文字列にマッチする正規表現を生成
kana_re = re.compile("^[ぁ-ゖ]+$")

# 単語分割用のオブジェクト生成
tagger = MeCab.Tagger(r'-d "C:\\Program Files (x86)\\MeCab\\dic\\ipadic" -u "C:\\Program Files (x86)\\MeCab\\dic\\NEologd\\NEologd.20200910-u.dic"')

# 単語分割しつつ、ひらがなのみの単語を除く
token_list = []
for i in range(0, len(df["text"])):
    # 単語分割
    parsed_lines = tagger.parse(df["text"][i]).split("\n")[:-2]
    surfaces = [l.split('\t')[0] for l in parsed_lines]
    #   # ひらがなのみの単語を除外しつつ, token_listに保存
    token_list.append([t for t in surfaces if not kana_re.match(t)])
#%%
# ストップワードリストの読み込み
f = open("C:/Users/r1ryo/Downloads/相馬ゼミ/Japanese.txt")
stopword = f.read()
f.close()
texts = []
stop_words = set(stopword.split())
# ストップワードリストに含まれないもののみをtextに格納
for i in range(0, len(df["text"])):
    texts.append([word for word in token_list[i] if word not in stop_words])

# それぞれの単語の出現回数を計測
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

# 出現回数100回未満の単語を除去
texts = [[token for token in text if frequency[token] > 99] for text in texts]

# 辞書の作成
dictionary = corpora.Dictionary(texts)
dictionary.filter_extremes(no_below=3, no_above=0.8)

# コーパスの作成
corpus = [dictionary.doc2bow(t) for t in texts]

# トピック数2から22まで動かし, それぞれ2指標の値を計算
start = 2
limit = 22
step = 1

coherence_vals = []
perplexity_vals = []

for n_topic in tqdm(range(start, limit, step)):
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=n_topic, random_state=0)
    perplexity_vals.append(np.exp2(-lda_model.log_perplexity(corpus)))
    coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_vals.append(coherence_model_lda.get_coherence())

# 計算結果を表示する
x = range(start, limit, step)
#%%
fig, ax1 = plt.subplots(figsize=(12,5))

# coherence
c1 = 'darkturquoise'
ax1.plot(x, coherence_vals, 'o-', color=c1)
ax1.set_xlabel('Num Topics')
ax1.set_ylabel('Coherence', color=c1); ax1.tick_params('y', colors=c1)

# perplexity
c2 = 'slategray'
ax2 = ax1.twinx()
ax2.plot(x, perplexity_vals, 'o-', color=c2)
ax2.set_ylabel('Perplexity', color=c2); ax2.tick_params('y', colors=c2)

ax1.set_xticks(x)
fig.tight_layout()
# png形式で保存
plt.savefig("Outputs/coherence_and_perplexity.png", format='png')

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=6, alpha='symmetric', random_state=0)

import matplotlib
print(matplotlib.matplotlib_fname())

fig, axs = plt.subplots(ncols=2, nrows=math.ceil(lda_model.num_topics/2), figsize=(16,20))
axs = axs.flatten()

def color_func(word, font_size, position, orientation, random_state, font_path):
    return 'darkturquoise'
### FONT_PATH = "「'ipeaxg.ttf'で終わるパス」"
for i, t in enumerate(range(lda_model.num_topics)):
    x = dict(lda_model.show_topic(t, 30))
    im = WordCloud(
        ### font_path=FONT_PATH,
        background_color='black',
        color_func=color_func,
        max_words=4000,
        width=300, height=300,
        random_state=0
    ).generate_from_frequencies(x)
    axs[i].imshow(im.recolor(colormap= 'Paired_r' , random_state=244), alpha=0.98)
    axs[i].axis('off')
    axs[i].set_title('Topic '+str(t))

plt.tight_layout()

plt.savefig("Outputs/wordcloud.png", format="png")
#%%
