import numpy as np
import pandas as pd
import pickle
import glob
#%%
# パスで指定したファイルの一覧をリスト形式で取得. （ここでは一階層下のtestファイル以下）
csv_files = glob.glob('D:/CSV_file_NYT_2011-2021/*.csv')

#読み込むファイルのリストを表示
for a in csv_files:
    print(a)

#csvファイルの中身を追加していくリストを用意
data_list = []

#読み込むファイルのリストを操作
for file in csv_files:
    data_list.append(pd.read_csv(file))
#%%
#リストを全て行方向に結合
#axis=0:行方向に結合, sort
df = pd.concat(data_list, sort=True, ignore_index=True)
#%%
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')

def clean_text(x):
    x = str(x)
    x = x.lower()
    x = re.sub(r'#[A-Za-z0-9]*', ' ', x)
    x = re.sub(r'https*://.*', ' ', x)
    x = re.sub(r'@[A-Za-z0-9]+', ' ', x)
    tokens = word_tokenize(x)
    x = ' '.join([w for w in tokens if not w.lower() in stop_words])
    x = re.sub(r'[%s]' % re.escape('!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~“…”’'), ' ', x)
    x = re.sub(r'\d+', ' ', x)
    x = re.sub(r'\n+', ' ', x)
    x = re.sub(r'\s{2,}', ' ', x)
    return x
#%%
from tqdm import tqdm
count_array = []

for i in tqdm(range(0, len(df.abstract))):
    if not pd.isna(df["abstract"][i]):
        count_array.append(i)
#%%
df = df.iloc[count_array, :]
#%%
df['clean_text'] = df.abstract.apply(clean_text)
df.head()
#%%
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
stop_words = stopwords.words('english')
print(stop_words)
#%%
from bertopic import BERTopic
abstract = df.clean_text.to_list()
timestamp = df.pub_date.to_list()
#%%
del df
del count_array
del csv_files
del data_list
del file
del stop_words
#%%
topic_model = BERTopic(language="english")
topics, probs = topic_model.fit_transform(abstract)
#%%
topic_model.get_topic_info()
#%%
visual_topic = topic_model.visualize_topics()
#%%
topic_model.visualize_barchart()
#%%
topics_over_time = topic_model.topics_over_time(abstract, topics, timestamp, nr_bins=20)
topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=20)
#%%
visual_topic.savefig("save.png")
#%%
import pickle
with open('../Outputs/save.png', 'wb') as p:
    pickle.dump(visual_topic, p)

#%%
import pickle
with open('../Outputs/save.png', 'rb') as p:
    l = pickle.load(p)
    print(l)
#%%

news_desk_list = ["Adventure Sports", "Arts & Leisure", "Arts", "Automobiles", "Blogs", "Books", "Booming",
                  "Business Day", "Business", "Cars", "Circuits", "Classifieds", "Connecticut", "Crosswords & Games",
                  "Culture", "DealBook", "Dining", "Editorial", "Education", "Energy", "Entrepreneurs",
                  "Environment", "Escapes", "Fashion & Style", "Fashion", "Favorites", "Financial, Flight", "Food",
                  "Foreign", "Generations", "Giving", "Global Home", "Health & Fitness", "Health", "Home & Garden",
                  "Home", "Jobs", "Key", "Letters", "Long Island", "Magazine", "Market Place", "Media",
                  "Men's Health", "Metro", "Metropolitan", "Movies", "Museums", "National", "Nesting", "Obits",
                  "Obituaries", "Obituary", "OpEd", "Opinion", "Outlook", "Personal Investing", "Personal Tech",
                  "Play", "Politics", "Regionals", "Retail", "Retirement", "Science", "Small Business", "Society",
                  "Sports", "Style", "Sunday Business", "Sunday Review", "Sunday Styles", "T Magazine", "T Style",
                  "Technology", "Teens", "Television", "The Arts", "The Business of Green", "The City Desk",
                  "The City", "The Marathon", "The Millennium", "The Natural World", "The Upshot", "The Weekend",
                  "The Year in Pictures", "Theater", "Then & Now", "Thursday Styles", "Times Topics", "Travel", "U.S.",
                  "Universal", "Upshot", "UrbanEye", "Vacation", "Washington", "Wealth", "Weather", "Week in Review",
                  "Week", "Weekend", "Westchester", "Wireless Living", "Women's Health", "Working", "Workplace",
                  "World", "Your Money"]
#%%
len(news_desk_list)
#%%
