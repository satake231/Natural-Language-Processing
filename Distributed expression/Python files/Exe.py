import os
import pickle
import csv
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import BertModel, BertJapaneseTokenizer
import re

# 対象となる単語
target_words = ['草']
# コーパスの年代
decade = 2014
# 各文章の長さ
sequence_length = 256
# 事前学習モデルの指定
pretrained_weights='cl-tohoku/bert-base-japanese-whole-word-masking'
# コーパスファイルのパス
coha_dir='data/coha'
# 一度に処理するデータ(文章)数
buffer_size=512
# 隠れ層の取り方(詳細は後述)
mode = 4
# ファイル読み込み, 出力のため
word = 'kusa'
# 分散表現のファイル出力先
outdir = 'data'
output_path = '{}/{}/usages_16_len{}_{}_12.dict'.format(outdir, word, sequence_length, decade)
tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_weights)

i2w = {}
print(target_words)
for t in target_words:
    t_id = tokenizer.encode(' '.join(target_words))[1]
    # tにターゲットワードが、t_idにそのトークンidが入る
    i2w[t_id] = t

# id化された文章
batch_input_ids = []
# 対象となる単語
batch_tokens = []
# トークン化された文章中における対象単語の位置
batch_pos = []
# 文章の冒頭と末尾に101と102を入れる前のid化された文章?
batch_snippets = []
# 文章の年代
batch_decades = []

usages = defaultdict(list)
#%%
tokenizer = BertJapaneseTokenizer.from_pretrained(pretrained_weights)
model = BertModel.from_pretrained(pretrained_weights)

def get_context(token_ids, target_position, sequence_length=128):
    """
    Given a text containing a target word, return the sentence snippet which surrounds the target word
    (and the target word's position in the snippet).

    :param token_ids: list of token ids (for an entire line of text)
    :param target_position: index of the target word's position in `tokens`
    :param sequence_length: desired length for output sequence (e.g. 128, 256, 512)
    :return: (context_ids, new_target_position)
                context_ids: list of token ids for the output sequence
                new_target_position: index of the target word's position in `context_ids`
    """
    # -2 as [CLS] and [SEP] tokens will be added later; /2 as it's a one-sided window
    window_size = int((sequence_length - 2) / 2)
    context_start = max([0, target_position - window_size])
    padding_offset = max([0, window_size - target_position])
    padding_offset += max([0, target_position + window_size - len(token_ids)])

    context_ids = token_ids[context_start:target_position + window_size]
    context_ids += padding_offset * [0]

    new_target_position = target_position - context_start

    return context_ids, new_target_position

def clean_text(text):
    # URLの削除
    replaced_text = re.sub(r'http\S+', '', text)
    # リプライなどの返信先を削除
    replaced_text = re.sub(r'@[w/:%#$&?()~.=+-…]+[:]? ', "", replaced_text)
    # 「RT」の文字列を削除
    replaced_text = re.sub(r"(^RT )", "", replaced_text)
    return replaced_text

# 2014_kusa.csvから文章を読み込む
with open('{}_{}.csv'.format(decade, word), 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    lines = []
    for row in reader:
        line = clean_text(row[0])
        lines.append(line)
        # コーパス内で重複している文章を削除する
        # twitterコーパスの場合には同じ文章が連続して入っていることがあった（ボット？）ので、これを追加した
        lines = list(set(lines))

# 各文章に対して処理
for L, line in enumerate(lines):
    # 形態素解析して分かち書き
    tokens = tokenizer.encode(line)
    # 各トークンごとに処理. tokenにはトークン, posは各トークンの位置を示す
    for pos, token in enumerate(tokens):
        # トークンがターゲットワードであれば実行
        if token in i2w:
            context_ids, pos_in_context = get_context(tokens, pos, sequence_length)
            # id化された文章の冒頭に101, 末尾に102を追加
            input_ids = [101] + context_ids + [102]

            # add usage info to buffers
            # 各文章をid化したものを入れるリスト
            batch_input_ids.append(input_ids)
            # ターゲットワードを入れるリスト. 今回であれば['草']になる
            batch_tokens.append(i2w[token])
            # 各文章におけるターゲットワードの位置
            batch_pos.append(pos_in_context)
            # id化した文章にSpecialTokenを追加したものを入れるリスト
            batch_snippets.append(context_ids)
            # 年代
            batch_decades.append(decade)

        # コーパス内のすべての文章、あるいはbuffer_sizeで指定したデータ(文章)数に対して処理を行ったら
        if (len(batch_input_ids) >= buffer_size) or (L == len(lines) - 1 and pos == len(tokens) - 1):
            print("=====================")
            print("{} data_size: {}".format(decade ,len(batch_input_ids)))
            print("=====================")
            with torch.no_grad():
                # collect list of input ids into a single batch tensor
                input_ids_tensor = torch.tensor(batch_input_ids)
                if torch.cuda.is_available():
                    input_ids_tensor = input_ids_tensor.to('cuda')
                # 隠れ層の出力. output_hidden_states = Trueにしないと出力されない
                outputs = model(input_ids_tensor, output_hidden_states = True)

                if torch.cuda.is_available():
                    hidden_states = [l.detach().cpu().clone().numpy() for l in outputs[2]]
                else:
                    hidden_states = [l.clone().numpy() for l in outputs[2]]

                # 得られたhidden_statesをnumpyのndarrayにして結合
                # 13(隠れ12層+最終層)×文章数×768次元になる
                hidden_states = np.stack(hidden_states)  # (13, B, |s|, 768)
                # 12層すべての和をとる
                if mode == 12:
                    usage_vectors = np.sum(hidden_states[1:, :, :, :], axis=0)
                # 最終4層のみの和をとる
                elif mode == 4:
                    usage_vectors = np.sum(hidden_states[-4:], axis=0)
# すでにファイルが存在すれば続きから追記
if output_path and os.path.exists(output_path):
    with open(output_path, 'rb') as f:
        usages = pickle.load(f)

# store usage tuples in a dictionary: lemma -> (vector, snippet, position, decade)
# 各文章の分usageをusagesに追加
for b in np.arange(len(batch_input_ids)):
    # 文章の分散表現
    # batch_pos[b]に+1されているのは、スペシャルトークンの分だけターゲットワードの位置が後ろに一個分ズレているから
    usage_vector = usage_vectors[b, batch_pos[b]+1, :]
    usages[batch_tokens[b]].append(
        (usage_vector, batch_snippets[b], batch_pos[b], batch_decades[b]))
batch_input_ids, batch_tokens, batch_pos, batch_snippets, batch_decades = [], [], [], [], []

# usagesを外部ファイルに出力
if output_path:
    # ファイルが無ければ作成する
    if not os.path.exists(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(usages, file=f)
#%%
