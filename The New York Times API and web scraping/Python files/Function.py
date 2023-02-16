# start_year から end_yearまでの記事を取得します。
# star_year = 2018, end_year=2020なら、2018, 2019, 2020の三年分の記事を取得します。
# 一分間あたりのリクエスト制限にはかからなようにしていますが、一日当たりの上限には対応していないので範囲指定を広くし過ぎるのには注意です。
# 具体的には、80年分以上を一日で取得するのは止めておきましょう。

import pandas as pd
import requests
from tqdm import tqdm
import time

def get_articles(start_year, end_year, your_api_key):
    url = "https://api.nytimes.com/svc/archive/v1/{}/{}.json?api-key={}"
    path = "./{}_{}.csv"

    for year in range(start_year, end_year + 1):
        for month in range(0, 12):
            api_endpoint = url.format(year, month + 1, your_api_key)

            res = requests.get(api_endpoint)

            res_json = res.json()

            df = pd.DataFrame({'abstract':[0],
                               'lead_paragraph':[0],
                               'pub_date':[0],
                               'print_headline':[0],
                               'main_headline':[0],
                               'keywords':[0],
                               'news_desk':[0],
                               'word_count':[0],
                               'snippet':[0],
                               'web_url':[0]}, index=[0])

            for i in tqdm(range(0, len(res_json['response']['docs']))):
                tmp = []
                for j in range(0, len(res_json['response']['docs'][i]['keywords'])):
                    tmp.append(res_json['response']['docs'][i]['keywords'][j]['value'])
                tmp = '*'.join(tmp)
                df.loc[i] = [res_json['response']['docs'][i]['abstract'], res_json['response']['docs'][i]['lead_paragraph'],
                             res_json['response']['docs'][i]['pub_date'], res_json['response']['docs'][i]['headline']['print_headline'],
                             res_json['response']['docs'][i]['headline']['main'], tmp, res_json['response']['docs'][i]['news_desk'],
                             res_json['response']['docs'][i]['word_count'], res_json['response']['docs'][i]['snippet'],
                             res_json['response']['docs'][i]['web_url']]

            df.to_csv(path.format(year, month + 1))

            time.sleep(6)
#%%
get_articles(1954, 1979, "XKJP6DlmzSwBe5jGv8BaDs4An3Cgo1k1")
#%%
