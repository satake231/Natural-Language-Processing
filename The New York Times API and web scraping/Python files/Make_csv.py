# ライブラリのインポート
import pandas as pd
import requests
from tqdm import tqdm

#%%
# エンドポイントへリクエスト
api_endpoint = "https://api.nytimes.com/svc/archive/v1/2021/12.json?api-key="

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
#%%
for i in tqdm(range(0, len(res_json['response']['docs']))):
    tmp = []
    for j in range(0, len(res_json['response']['docs'][i]['keywords'])):
        tmp.append(res_json['response']['docs'][i]['keywords'][j]['value'])
    tmp = ' '.join(tmp)
    df.loc[i] = [res_json['response']['docs'][i]['abstract'], res_json['response']['docs'][i]['lead_paragraph'],
                 res_json['response']['docs'][i]['pub_date'], res_json['response']['docs'][i]['headline']['print_headline'],
                 res_json['response']['docs'][i]['headline']['main'], tmp, res_json['response']['docs'][i]['news_desk'],
                 res_json['response']['docs'][i]['word_count'], res_json['response']['docs'][i]['snippet'],
                 res_json['response']['docs'][i]['web_url']]
#%%
df.to_csv("./2021_12.csv")

#%%


