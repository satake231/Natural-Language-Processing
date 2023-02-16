# ライブラリのインポート
import pandas as pd
import requests
from tqdm import tqdm
#%%
# エンドポイントへリクエスト
# "2020/1"部分が年月の指定になっている
api_endpoint = "https://api.nytimes.com/svc/archive/v1/2020/4.json?api-key=XKJP6DlmzSwBe5jGv8BaDs4An3Cgo1k1"

res = requests.get(api_endpoint)

print(res)
#%%
# 取得したJSONオブジェクトを辞書形式に変換
res_json = res.json()
print(res_json)
#%%
print(res_json.keys())
# copyright, response
#%%
print(res_json['copyright'])
#%%
# response 内のキーの一覧を表示
print(res_json['response'].keys())
# docs, meta
#%%
print(res_json['response']['docs'])
# object type: list
#%%
# 記事のヒット数
print(res_json['response']['meta'])
# hits : 4480
#%%
# docs 内のキーの一覧を表示
n = 0
print(res_json['response']['docs'][n].keys())
# abstract, lead_paragraph, headline, keywords, news_desk, pub_date
#%%
# サイトのURL
print(res_json['response']['docs'][n]['web_url'])
#%%
# abstract: 記事の見出しの後に続く、本文の概要を纏めたところ
print(res_json['response']['docs'][n]['abstract'])
#%%
# lead_paragraph: 恐らく、本文の一番最初（有料会員ではないため確認できない）
print(res_json['response']['docs'][n]['lead_paragraph'])
#%%
# pub_date: 記事の発行日
print(res_json['response']['docs'][n]['pub_date'])
#%%
# print_headline: 謎。サイト上ではこれにあたるheadlineを確認できず
print(res_json['response']['docs'][n]['headline']['print_headline'])
#%%
# main: サイト上において表示されている「見出し」部分
print(res_json['response']['docs'][n]['headline']['main'])
#%%
# keywords: 記事によって数が異なる。keywordがない記事もある
print(res_json['response']['docs'][n]['keywords'][0]['value'])
print(len(res_json['response']['docs'][n]['keywords']))
#%%
# news_desk: 記事の属するジャンル
print(res_json['response']['docs'][n]['news_desk'])
#%%
# word_count: 記事全体の文字数
print(res_json['response']['docs'][n]['word_count'])
#%%
# snippet: 和訳「抜粋」　こちらも会員ではないため中身が実際の記事の何に当たるかが確認できず
print(res_json['response']['docs'][n]['snippet'])
#%%
# 格納用データフレームの製作
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
    df.loc[i] = [res_json['response']['docs'][i]['abstract'], res_json['response']['docs'][i]['lead_paragraph'],
                 res_json['response']['docs'][i]['pub_date'], res_json['response']['docs'][i]['headline']['print_headline'],
                 res_json['response']['docs'][i]['headline']['main'], tmp, res_json['response']['docs'][i]['news_desk'],
                 res_json['response']['docs'][i]['word_count'], res_json['response']['docs'][i]['snippet'],
                 res_json['response']['docs'][i]['web_url']]
#%%
df.to_csv("./iris1.csv")
#%%
