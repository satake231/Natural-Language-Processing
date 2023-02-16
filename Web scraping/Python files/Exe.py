import time
# 現在時刻を取得したり、処理を一時停止できる。
import requests
from bs4 import BeautifulSoup
def parse_url(url, sleep_second=1):
    # 引数にとったURLからGETリクエストを行い、webデータを取得
    res = requests.get(url)
    # サーバーに負荷をかけないようにインターバルを設ける。
    time.sleep(sleep_second)
    # HTMLの解析
    return BeautifulSoup(res.content, "html.parser")

URL = "https://coco.to/movies"
# htmlの一部表示
parse_url(URL).title
# <title>2022年公開の映画</title>が出力される。
#%%
soup = parse_url(URL)
# ページのタイトルを参照(<title>タグの部分)
print(soup.title)
# タグで囲まれたテキスト部分だけを抽出
print(soup.title.text)
# ページ内のli要素を1つ抽出
print(soup.find("li").text)
#%%
def should_crawl(score, upper_score=90, lower_score=10):
    # 10%以上90%以下のスコアのみを抽出
    return lower_score <= int(score.replace("％", "")) <= upper_score

# 収集先のベースurl
domain = "https://coco.to/"
prefix = "movies/%s"
# 収集する映画の起点となる年
current_year = 2022
# 収集する年数
n = 15
# 収集したurlを入れる配列
checked_urls = list()
# 2008-2022の映画を取得
for i in range(n):
    # 20XX年の映画一覧ページのurl
    movie_list_url = domain + prefix % (current_year - i)
    # htmlのパース
    soup = parse_url(movie_list_url)
    print("抽出開始", movie_list_url)

    # 総合評価やリンクが含まれている親要素を抽出
    for panel in soup.find_all("div", class_="li_panel"):
        # li_txtクラスに属しているdiv要素を抜き出す(総合評価の部分)
        score = panel.find("div", class_="li_txt").text
        # 評価が10%以上90%以上の映画のurlだけを抜き出す
        if should_crawl(score):
            # a要素に含まれるurlを抽出(映画の個別url)
            uri = panel.find("a")["href"]
            # checked_urlsに追加
            checked_urls.append(domain + uri)
    # 抽出が終わった年のurlと収集した累計url数を出力する
    print("抽出終了", movie_list_url, "取得済みURL", len(checked_urls))
print("先頭のひとつを表示",checked_urls[0])
#%%
# プログレスバーの表示用
from tqdm import tqdm

# データ本体
crawled = list()
# checked_urlsのうちどのくらい処理が終わったかをプログレスバーで表示する
for url in tqdm(checked_urls):
    # データの取得
    soup = parse_url(url)
    # つぶやき本文
    # (div要素のtweet_text clearflt clearbothクラスに囲まれた部分)を抽出
    texts = soup.find_all("div", class_="tweet_text clearflt clearboth")
    # 「良い」「普通」「残念」の評価
    # (span要素のjudge_textクラスに囲まれた部分)を抽出
    labels = soup.find_all("span", class_="judge_text")
    # 映画のタイトルを抽出
    title = soup.find("div", class_="title_").find("h1")
    # データのパース
    for t, l in zip(texts, labels):
        text = t.get_text(strip=True)
        # span要素を取り除く(「いいね」のテキスト部分)
        metadata = t.find("span").get_text(strip=True)
        stripped_text = text.replace(metadata, "").strip()
        # つぶやき本文、評価、映画のタイトルを1つのデータにする
        article = {
            "text": stripped_text,
            "label": l.get_text(strip=True),
            "title": title.get_text(strip=True)
        }
        crawled.append(article)
print("全URLの取得完了")
#%%
# データのパース
for t, l in zip(texts, labels):
    text = t.get_text(strip=True)
    # span要素を取り除く(「いいね」のテキスト部分)
    metadata = t.find("span").get_text(strip=True)
    stripped_text = text.replace(metadata, "").strip()
    # つぶやき本文、評価、映画のタイトルを1つのデータにする
    article = {
        "text": stripped_text,
        "label": l.get_text(strip=True),
        "title": title.get_text(strip=True)
    }
    crawled.append(article)
#%%
import pandas
df = pandas.DataFrame(crawled)
# カラムごと150文字まで表示
pandas.set_option('display.max_colwidth',150)
# text, label, titleのカラムを持ったdataframe型に変換
display(df)
#%%
# 訓練データに使うデータを「良い」と「残念」の評価だけに絞る
filtered_by_label = df.query("label == '良い' | label == '残念'")
display(filtered_by_label)
#%%

# 「良い」「残念」のラベルでグルーピング
group_by_label = filtered_by_label.groupby("label")
print(type(group_by_label))
# 各ラベルのデータ数を出力
labels_size = group_by_label.size()
display(labels_size)
#%%
# label(評価)のうち数が少ないほうの総数。今回であれば「残念」の評価数

n = labels_size.min()
# グルーピングしたデータフレームに対して、それぞれラベル付けされたデータ
# をnの数だけ抽出
#　n はサイズ
dataset = group_by_label.apply(lambda x: x.sample(n, random_state=0))
#random_state = 0 は乱数を固定。（いつ何時どこの環境からでも同じ抽出結果になる）
display(dataset)
print(type(dataset))
#%%
from sklearn.preprocessing import LabelEncoder
label_vectorizer = LabelEncoder()
# 「残念」を0に、「良い」を1に置き換える
# ラベルを数値変換
transformed_label = label_vectorizer.fit_transform(dataset.get("label"))
print(transformed_label)
print(len(transformed_label))
# 元のデータセットのラベル部分を上書き
dataset["label"] = transformed_label
#%%
from sklearn.model_selection import train_test_split
# 入力と出力に分割
# 入力(x)はツイートの文章データ, 出力(y)は「良い」「残念」の評価
x, y = dataset.get("text"), dataset.get("label")
# 学習とテストデータに9:1で分割
X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=0.1, stratify=y, random_state=0)
# それぞれの数があっているか確認
print([len(c) for c in [X_train, X_test, y_train, y_test]])
#%%
from janome.tokenizer import Tokenizer
# 分かち書きで出力
tokenizer = Tokenizer(wakati=True)
#%%
from sklearn.feature_extraction.text import CountVectorizer
# ベクトル化の準備
feature_vectorizer = CountVectorizer(binary=True, analyzer=tokenizer.tokenize)
#%%
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
# 文章のベクトル化
transformed_X_train = feature_vectorizer.fit_transform(X_train)
# 訓練データをもとに学習
classifier.fit(transformed_X_train, y_train)
#%%
from sklearn.metrics import classification_report
# テストデータの文章をベクトル化
vectorized = feature_vectorizer.transform(X_test)
# ベクトル化された文章テストデータを用いて予測
y_pred = classifier.predict(vectorized)
# モデルの評価指標をまとめて出力
print(classification_report(y_test, y_pred,target_names=label_vectorizer.classes_))
#%%
from sklearn.metrics import confusion_matrix
cm_1 = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm_1)
#%%
# モデルの分析
from pandas import Series
# 単語をkey, その単語が判定に及ぼした重みをvalueにとる辞書
feature_to_weight = dict()
# classifierのcoefに各単語の重みが格納されている
for w, name in zip(classifier.coef_[0], feature_vectorizer.get_feature_names()):
    feature_to_weight[name] = w
se = Series(feature_to_weight)
se.sort_values(ascending=False, inplace=True)
print("Positive or Negative")
print("--Positiveの判定に効いた素性")
print(se[:20])
print("--Negativeの判定に効いた素性")
print(se[-20:])
print("--" * 50)
#%%
def validate():
    # 学習
    classifier = LogisticRegression()
    transformed_X_train = feature_vectorizer.fit_transform(X_train)
    classifier.fit(transformed_X_train, y_train)
    # 評価
    vectorized = feature_vectorizer.transform(X_test)
    y_pred = classifier.predict(vectorized)
    print(classification_report(y_test, y_pred))
    # モデルの分析
    feature_to_weight = dict()
    for w, name in zip(classifier.coef_[0],
                       feature_vectorizer.get_feature_names()):
        feature_to_weight[name] = w
    se = Series(feature_to_weight)
    se.sort_values(ascending=False, inplace=True)
    print("--Positiveの判定に効いた素性")
    print(se[:20])
    print("--Negativeの判定に効いた素性")
    print(se[-20:])
    print("--" * 50)
#%%
from janome.analyzer import Analyzer
from janome.charfilter import RegexReplaceCharFilter
from janome.tokenfilter import ExtractAttributeFilter, POSKeepFilter, TokenFilter
# 前処理
# 「」や『』で囲まれた映画のタイトルを除去
# URLを除去
char_filters = [
    RegexReplaceCharFilter("^[『「【].*[』」】]", ""),
    RegexReplaceCharFilter("(https?:\/\/[\w\.\-/:\#\?\=\&\;\%\~\+]*)", "")]
# 後処理
# 名詞, 動詞, 形容詞, 副詞だけを対象にする
# 表記ゆれを無くすため原形へ修正する
token_filters = [
    POSKeepFilter(['名詞', '動詞', '形容詞', '副詞']),
    ExtractAttributeFilter("base_form")]
# Tokenizerの再初期化
tokenizer = Tokenizer()
# 前処理・後処理が追加されたVectorizerに変更
analyzer = Analyzer(char_filters=char_filters,
                    tokenizer=tokenizer, token_filters=token_filters)
feature_vectorizer = CountVectorizer(binary=True, analyzer=analyzer.analyze)
# 再評価
result = validate()
#%%
# 検証用のDataFrameを作成
validate_df = pandas.concat([X_test, y_test], axis=1)
# 予測した評価の列を追加
validate_df["y_pred"] = result
# 予測とラベルが異なるものを抽出
false_positive = validate_df.query("y_pred == 1 & label == 0")
display(false_positive)
#%%
