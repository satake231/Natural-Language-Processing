#coding=UTF-8
# 形態素解析の勉強をしよう
#%%
# 1. 形態素解析とは
# 形態素解析: 文章を最小単位に分割して品詞や活用形を判別すること
#%% Mecabを用いた形態素解析
# Mecab: 日本語用の形態素解析ツール
import MeCab

Tagger = MeCab.Tagger()
# NEologd使用時は、カッコ内にr'-d "C:\\Program Files (x86)\\MeCab\\dic\\ipadic" -u "C:\\Program Files (x86)\\MeCab\\dic\\NEologd\\NEologd.20200910-u.dic"'
result = Tagger.parse('形態素解析を学ぶということは、形態素を解析するということです。')
#parseメソッドは解析したい文字を引数にとる
print(result)
#%%
tgr = MeCab.Tagger('-O wakati')
# -o : 出力フォーマット指定
# wakachi : 分かち書きの状態で出力
r = tgr.parse('今日はいい天気だった。')
print(r)
#%%
# 辞書の指定
Tagger_uni = MeCab.Tagger('-d c:/users/r1ryo/anaconda3/lib/site-packages/unidic_lite/dicdir')
# '-d (辞書のパス)' の形式で引数に辞書を任意に指定できる
result_uni = Tagger_uni.parse('朝ごはんを抜きにしてからもう一年だ')
print(result_uni)
#%%
import pandas as pd

sentence = """吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。
第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢あったがこんな片輪には一度も出会わした事がない。
のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。
どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。"""
#%%
from io import StringIO
tagger = MeCab.Tagger()
# StringIO : 文字列をファイルライクに扱える
buffer = StringIO(tagger.parse(sentence))
# str型のbufferをdataframe型に変換
df = pd.read_csv(
    buffer,
    names = ["表層形", "発音", "読み", "原形", "品詞", "品詞再分類1", "品詞細分類2", "品詞細分類3", "活用型", "活用形"],
    skipfooter=1,
    sep="[\t, , ]",
    engine="python",
)
#%%
noun_df = df.query('品詞.str.contains("名詞")')
#%%
# 品詞が助動詞となっているデータを抽出
print(df[df['品詞']=="助動詞"])
#%%
# 品詞が助動詞となっているデータを抽出
df.query('品詞=="助動詞"')
#%%
# 「原形」の列だけを抽出
print(df["原形"])
#%%
# 特定の種類の単語について、出現回数を辞書形式で保存する
noun_counter = dict()

for word in noun_df["表層形"]:
    # 2.
    if noun_counter.get(word):
        noun_counter[word] += 1
    # 1.
    else:
        noun_counter[word] = 1

print(noun_counter)
#%%
sorted_dic = sorted(noun_counter.items(), key=lambda x: x[1], reverse=True)
print(sorted_dic)
#%%
# 上位10個出力
print(sorted(noun_counter.items(), key=lambda x: x[1], reverse=True)[:10])
#%%

# 坊ちゃんの形態素解析
sentence = """親譲りの無鉄砲で小供の時から損ばかりしている。小学校に居る時分学校の二階から飛び降りて一週間ほど腰を抜かした事がある。なぜそんな無闇をしたと聞く人があるかも知れぬ。別段深い理由でもない。新築の二階から首を出していたら、同級生の一人が冗談に、いくら威張っても、そこから飛び降りる事は出来まい。弱虫やーい。と囃したからである。小使に負ぶさって帰って来た時、おやじが大きな眼をして二階ぐらいから飛び降りて腰を抜かす奴があるかと云ったから、この次は抜かさずに飛んで見せますと答えた。
親類のものから西洋製のナイフを貰って奇麗な刃を日に翳して、友達に見せていたら、一人が光る事は光るが切れそうもないと云った。切れぬ事があるか、何でも切ってみせると受け合った。そんなら君の指を切ってみろと注文したから、何だ指ぐらいこの通りだと右の手の親指の甲をはすに切り込んだ。幸ナイフが小さいのと、親指の骨が堅かったので、今だに親指は手に付いている。しかし創痕は死ぬまで消えぬ。
庭を東へ二十歩に行き尽すと、南上がりにいささかばかりの菜園があって、真中に栗の木が一本立っている。これは命より大事な栗だ。実の熟する時分は起き抜けに背戸を出て落ちた奴を拾ってきて、学校で食う。菜園の西側が山城屋という質屋の庭続きで、この質屋に勘太郎という十三四の倅が居た。勘太郎は無論弱虫である。弱虫の癖に四つ目垣を乗りこえて、栗を盗みにくる。ある日の夕方折戸の蔭に隠れて、とうとう勘太郎を捕まえてやった。その時勘太郎は逃げ路を失って、一生懸命に飛びかかってきた。向うは二つばかり年上である。弱虫だが力は強い。鉢の開いた頭を、こっちの胸へ宛ててぐいぐい押した拍子に、勘太郎の頭がすべって、おれの袷の袖の中にはいった。邪魔になって手が使えぬから、無暗に手を振ったら、袖の中にある勘太郎の頭が、右左へぐらぐら靡いた。しまいに苦しがって袖の中から、おれの二の腕へ食い付いた。痛かったから勘太郎を垣根へ押しつけておいて、足搦をかけて向うへ倒してやった。山城屋の地面は菜園より六尺がた低い。勘太郎は四つ目垣を半分崩して、自分の領分へ真逆様に落ちて、ぐうと云った。勘太郎が落ちるときに、おれの袷の片袖がもげて、急に手が自由になった。その晩母が山城屋に詫びに行ったついでに袷の片袖も取り返して来た。
この外いたずらは大分やった。大工の兼公と肴屋の角をつれて、茂作の人参畠をあらした事がある。人参の芽が出揃わぬ処へ藁が一面に敷いてあったから、その上で三人が半日相撲をとりつづけに取ったら、人参がみんな踏みつぶされてしまった。古川の持っている田圃の井戸を埋めて尻を持ち込まれた事もある。太い孟宗の節を抜いて、深く埋めた中から水が湧き出て、そこいらの稲にみずがかかる仕掛であった。その時分はどんな仕掛か知らぬから、石や棒ちぎれをぎゅうぎゅう井戸の中へ挿し込んで、水が出なくなったのを見届けて、うちへ帰って飯を食っていたら、古川が真赤になって怒鳴り込んで来た。たしか罰金を出して済んだようである。
おやじはちっともおれを可愛がってくれなかった。母は兄ばかり贔屓にしていた。この兄はやに色が白くって、芝居の真似をして女形になるのが好きだった。おれを見る度にこいつはどうせ碌なものにはならないと、おやじが云った。乱暴で乱暴で行く先が案じられると母が云った。なるほど碌なものにはならない。ご覧の通りの始末である。行く先が案じられたのも無理はない。ただ懲役に行かないで生きているばかりである。
母が病気で死ぬ二三日前台所で宙返りをしてへっついの角で肋骨を撲って大いに痛かった。母が大層怒って、お前のようなものの顔は見たくないと云うから、親類へ泊りに行っていた。するととうとう死んだと云う報知が来た。そう早く死ぬとは思わなかった。そんな大病なら、もう少し大人しくすればよかったと思って帰って来た。そうしたら例の兄がおれを親不孝だ、おれのために、おっかさんが早く死んだんだと云った。口惜しかったから、兄の横っ面を張って大変叱られた。
母が死んでからは、おやじと兄と三人で暮していた。おやじは何にもせぬ男で、人の顔さえ見れば貴様は駄目だ駄目だと口癖のように云っていた。何が駄目なんだか今に分らない。妙なおやじがあったもんだ。兄は実業家になるとか云ってしきりに英語を勉強していた。元来女のような性分で、ずるいから、仲がよくなかった。十日に一遍ぐらいの割で喧嘩をしていた。ある時将棋をさしたら卑怯な待駒をして、人が困ると嬉しそうに冷やかした。あんまり腹が立ったから、手に在った飛車を眉間へ擲きつけてやった。眉間が割れて少々血が出た。兄がおやじに言付けた。おやじがおれを勘当すると言い出した。"""
#%%
#
def Mcb_str2df(str, dict_path):
    if len(dict_path) == 2:
        Tagger = MeCab.Tagger()
    else:
        Tagger = MeCab.Tagger(dict_path)

    buffer = StringIO(Tagger.parse(str))
# str型のbufferをdataframe型に変換
    df = pd.read_csv(
        buffer,
        names = ["表層形", "発音", "読み", "原形", "品詞", "品詞再分類1", "品詞細分類2", "品詞細分類3", "活用型", "活用形"],
        skipfooter=1,
        sep="[\t, , ]",
        engine="python",
    )

    return df

df_n = Mcb_str2df(sentence, 'no')

qrd_df = df_n.query('品詞.str.contains("名詞")')
#%%
counter = dict()

for word in qrd_df["表層形"]:
    if counter.get(word):
        counter[word] += 1
    else:
        counter[word] = 1

print(sorted(noun_counter.items(), key=lambda x: x[1], reverse=True)[:20])
#%% Janomeを用いた形態素解析
from janome.tokenizer import Tokenizer

t = Tokenizer()

sentence = """吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。
第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢あったがこんな片輪には一度も出会わした事がない。
のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。
どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。"""

# Mecabと同様のやりかたでは出力できない
# 全体を見るにはfor文で中身から引っ張って来る必要がある
for token in t.tokenize(sentence):
    print(token)
#%%
# 一部を指定して見たいときは"__next__()"を使用
token = t.tokenize(sentence).__next__()
print(token)
#%%
# surface
# 表層形‐→そのままの形
print(token.surface)
#%%
# part_of_speech
# 品詞 (品詞, 品詞細分類1, 品詞細分類2, 品詞細分類3)
# "*" は未定義の意
print(token.part_of_speech)
#%%
# infl_type
# 活用型
print(token.infl_type)
#%%
# infl_form
# 活用形
print(token.infl_form)
#%%
# base_form
# 基本形
print(token.base_form)
#%%
# reading
# 読み
print(token.reading)
#%%
# phonetic
# 発音
print(token.phonetic)
#%%
# 分かち書きも可能
t_wkt = Tokenizer(wakati=True)
# この場合はリスト指定すれば出力可能
print(list(t_wkt.tokenize(sentence)))
#%%
# リスト内包表記による属性指定されたリスト作成
# "[token.指定したい属性 for token in t.tokenize(sentence)]"
# 表層形を指定してみる
print([token.surface for token in t.tokenize(sentence)])
#%%
# リスト内包表記による品詞指定されたリスト作成
# "[token.指定したい属性 for token in t.tokenize(sentence)
#     if token.part_of_speech.startswith('品詞')]"
print([token.surface for token in t.tokenize(sentence)
       if token.part_of_speech.startswith('動詞')])
#%%
# 単語の出現回数をカウント

# collectionモジュールを使用
import collections

count = collections.Counter(t.tokenize(sentence, wakati=True))

# 辞書形式で出力
print(count)
#%%
# 前処理・後処理用モジュール

from janome.analyzer import Analyzer
from janome.charfilter import *
from janome.tokenfilter import *

# フィルタリング
# 正規表現とHTMLタグの消去
char_filter = [UnicodeNormalizeCharFilter(),
               RegexReplaceCharFilter('<.*?>', '')]

# 品詞指定抽出と小文字化、属性指定
token_filter = [POSKeepFilter(['名詞']),
                LowerCaseFilter(),
                ExtractAttributeFilter('surface')]

# 処理を施したものを出力
a = Analyzer(char_filters=char_filter, token_filters=token_filter)

for token in a.analyze(sentence):
    print(token)
#%%
# 複合名詞化
s = '複合名詞化を用いた形態素解析という自然言語処理'

a = Analyzer(token_filters=[CompoundNounFilter()])

for token in a.analyze(s):
    print(token)
#%%
# Analyzerによる単語出現回数のカウント
# TokenCountFilter() の使用によってカウント可能。TokenCountFilterは末尾に置くこと。
sentence = """吾輩は猫である。名前はまだ無い。
どこで生れたかとんと見当がつかぬ。何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。
吾輩はここで始めて人間というものを見た。しかもあとで聞くとそれは書生という人間中で一番獰悪な種族であったそうだ。
この書生というのは時々我々を捕えて煮て食うという話である。しかしその当時は何という考もなかったから別段恐しいとも思わなかった。
ただ彼の掌に載せられてスーと持ち上げられた時何だかフワフワした感じがあったばかりである。
掌の上で少し落ちついて書生の顔を見たのがいわゆる人間というものの見始であろう。この時妙なものだと思った感じが今でも残っている。
第一毛をもって装飾されべきはずの顔がつるつるしてまるで薬缶だ。その後猫にもだいぶ逢あったがこんな片輪には一度も出会わした事がない。
のみならず顔の真中があまりに突起している。そうしてその穴の中から時々ぷうぷうと煙を吹く。
どうも咽せぽくて実に弱った。これが人間の飲む煙草というものである事はようやくこの頃知った。"""

a = Analyzer(token_filters=[CompoundNounFilter(), TokenCountFilter()])

# listやdictなどの形式で指定して出力できる
list(a.analyze(sentence))
#%%
dict(a.analyze(sentence))
#%%
# 参考：https://note.nkmk.me/python-janome-tutorial/