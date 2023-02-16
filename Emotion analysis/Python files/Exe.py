import csv
np_dic = {}

fp = open("../../../Natural-Language-Processing/Emotion analysis/Database/pn.csv.m3.120408.trim", "rt", encoding="utf-8")
reader = csv.reader(fp, delimiter="\t")
for i, row in enumerate(reader):
    name = row[0]
    result = row[1]
    np_dic[name] = result
    if i % 1000 == 0: print(i)
print("ok")
#%%
print(np_dic["激怒"])
print(np_dic["喜び"])
print(np_dic["時間"])
#%%
from janome.tokenizer import Tokenizer

tokenizer = Tokenizer()
def np_rate(input_str):
    pos_count = 0
    neg_count = 0
    word_count = 0
    tokens = tokenizer.tokenize(input_str)
    for token in tokens:
        base_form = token.base_form
        if base_form in np_dic:
            if np_dic[base_form] == "p" :
                pos_count += 1
                print("POS:" + base_form)
            if np_dic[base_form] == "n" :
                neg_count += 1
                print("NEG:" + base_form)
        word_count += 1
    return pos_count, neg_count, word_count

print(np_rate("メロスは激怒した。必ず、かの邪智暴虐の王を除かねばならぬと決意した。"))
#%%
import re
import zipfile
import urllib.request
import os.path, glob
#%%
def get_flat_text_from_aozora(zip_url):
    zip_file_name = re.split(r"/", zip_url)[-1]
    print(zip_file_name)

    if not os.path.exists(zip_file_name):
        print("Download URL = ",zip_url)
        data = urllib.request.urlopen(zip_url).read()
        with open(zip_file_name, mode="wb") as f :
            f.write(data)
    else:
        print("Maybe already exists")

        dir,ext = os.path.splitext(zip_file_name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        unzipped_data = zipfile.ZipFile(zip_file_name, "r")
        unzipped_data.extractall(dir)
        unzipped_data.close()
        wild_path = os.path.join(dir, "*.txt")
        txt_file_path = glob.glob(wild_path)[0]

        print(txt_file_path)

        binary_data = open(txt_file_path, "rb").read()
        main_text = binary_data.decode("shift_jis")

        return main_text
#%%
import re
import urllib.request

url = 'http://x0213.org/codetable/jisx0213-2004-std.txt'
response = urllib.request.urlopen(url)

with response as f:
    content = f.read().decode('utf-8')
    ms = (re.match(r'(\d-\w{4})\s+U\+(\w{4,5})',l)for l in content.splitlines() if l[0] != "#")
    gaiji_table = {m[1]: chr(int(m[2], 16)) for m in ms if m}

def get_gaiji(s):
    m = re.search(r'第(\d)水準\d-(\d{1,2})-(\d{1,2})', s)
    if m:
        key = f'{m[1]}-{int(m[2])+32:2X}{int(m[3])+32:2X}'
        return gaiji_table.get(key, s)

    m = re.search(r"U\+(\w{4})", s)
    if m:
        return chr(int(m[1], 16))

    m = re.search(r'.*?(\d)-(\d{1,2})-(\d{1,2})', s)
    if m:
        key = f"{int(m[1])+2}-{int(m[2])+32:2X}{int(m[3])+32:2X]}"
        return gaiji_table.get(key,s)
    return s
#%%
def flatten_aozora(text):
    text = re.sub(r'※ [# .+?]', lambda m: get_gaiji(m[0]), text)
    text = re.split(r'\-{5,}', text)[2]
    text = re.split(r"底本:", text)[0]
    text = re.sub(r"<<.+?>>", "", text)
    text = re.sub(r"[#.+?]", "", text)
    text = text.strip()
    return text
#%%
import time
def get_all_flat_text_from_zip_list(zip_list):
    all_flat_text = ""
    for zip_url in zip_list:
        try:
            aozora_dl_text = get_flat_text_from_aozora(zip_url)
            flat_text = flatten_aozora(aozora_dl_text)
            all_flat_text += flat_text + ("\n")
            print(zip_url+": 取得&加工完了")
        except:
            import traceback
            traceback.print_exc()
            print(zip_url +"取得or解凍エラーのためスキップ")
        time.sleep(1)
    return all_flat_text
#%%
ZIP_URL = "https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip" #走れメロスのテキストデータ

aozora_dl_text = get_flat_text_from_aozora(ZIP_URL)
flat_text = flatten_aozora(aozora_dl_text)
print(flat_text[0:1000])
#%%
import matplotlib.pyplot as plt

ZIP_URL = "https://www.aozora.gr.jp/cards/000035/files/1567_ruby_4948.zip" #走れメロスのテキストデータ
aozora_dl_text = get_flat_text_from_aozora(ZIP_URL)
flat_text = flatten_aozora(aozora_dl_text)
mero_list = flat_text.split("\n")

x = []
y1 = []
y2 = []

total_word_count = 0

for mero_str in mero_list:
    pos_count, neg_count, word_count = np_rate(mero_str)

    if word_count < 1 :
        continue

    y1.append(pos_count/word_count)
    y2.append(neg_count/word_count)
    total_word_count += word_count
    x.append(total_word_count)

plt.plot(x, y1, marker = "o", color = "red", linestyle = "--", label = "positive" )
plt.plot(x, y2, marker = "x", color = "blue", linestyle = ":" , label = "negative")
plt.legend()
plt.savefig('../Outputs/melos.png', format='png')
plt.show()
#%%
ZIP_URL_1 = "https://www.aozora.gr.jp/cards/000879/files/127_ruby_150.zip"   #羅生門
aozora_dl_text_1 = get_flat_text_from_aozora(ZIP_URL_1)
flat_text_1 = flatten_aozora(aozora_dl_text_1)
mero_list_1 = flat_text_1.split("\n")

x_1 = []
y1_1 = []
y2_1 = []

total_word_count_1 = 0

for mero_str_1 in mero_list_1:
    pos_count_1, neg_count_1, word_count_1 = np_rate(mero_str_1)

    if word_count_1 < 1 :
        continue

    y1_1.append(pos_count_1/word_count_1)
    y2_1.append(neg_count_1/word_count_1)
    total_word_count_1 += word_count_1
    x_1.append(total_word_count_1)

plt.plot(x_1, y1_1, marker = "o", color = "red", linestyle = "--" )
plt.plot(x_1, y2_1, marker = "x", color = "blue", linestyle = ":" )
plt.savefig('../Outputs/rasho.png', format='png')
#%%
ZIP_URL_3 = "https://www.aozora.gr.jp/cards/000148/files/773_ruby_5968.zip" #こころ
aozora_dl_text_3 = get_flat_text_from_aozora(ZIP_URL_3)
flat_text_3 = flatten_aozora(aozora_dl_text_3)
mero_list_3 = flat_text_3.split("\n")

x_3 = []
y1_3 = []
y2_3 = []

total_word_count_3 = 0

for mero_str_3 in mero_list_3:
    pos_count_3, neg_count_3, word_count_3 = np_rate(mero_str_3)

    if word_count_3 < 1 :
        continue

    y1_3.append(pos_count_3/word_count_3)
    y2_3.append(neg_count_3/word_count_3)
    total_word_count_3 += word_count_3
    x_3.append(total_word_count_3)

plt.plot(x_3, y1_3, marker = "o", color = "red", linestyle = "--" )
plt.plot(x_3, y2_3, marker = "x", color = "blue", linestyle = ":" )
plt.savefig('../Outputs/cocoro.png', format='png')
#%%
