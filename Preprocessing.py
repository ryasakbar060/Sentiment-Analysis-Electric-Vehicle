import streamlit as st
import pandas as pd
import re
import emoji
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 
from openpyxl import Workbook
import base64

# Fungsi untuk cleansing teks
def cleansing(Text):
    Text = re.sub(r'RT', '', Text) # remove RT
    Text = Text.replace("<br>", " ") # remove <br>
    Text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)","",Text).split())
    Text = Text.lower() # Mengubah menjadi huruf kecil
    Text = re.sub(r"[^a-zA-Z0-9]", " ", Text)
    Text = emoji.demojize(Text)  # Menghilangkan emoji
    Text = re.sub(r':[a-zA-Z_]+:', '', Text)
    Text = re.sub(r'[^\w\s]', '', Text) # remove tanda baca
    Text = re.sub(r'\d+', '', Text) # remove angka
    Text = Text.replace("http://", " ").replace("https://", " ") # remove http
    Text = Text.replace('\\t'," ").replace('\\n', " ").replace('\\u', " ").replace('\\'," ")
    return Text

# Fungsi untuk normalisasi teks
def slang_normalization(text):
    df_slang = pd.read_excel("normalisasi.xlsx")
    slang_dict = dict(zip(df_slang['original'], df_slang['replacement']))
    text = ' '.join([slang_dict[word] if word in slang_dict else word for word in text.split()])
    return text

# Fungsi untuk tokenize teks
def tokenization(Text):
    tokens = word_tokenize(Text)
    return tokens

# Fungsi untuk tokenize teks
def remove_stopwords(tokens):
    list_stopwords = nltk.corpus.stopwords.words('indonesian')
    list_stopwords.extend(['yg', 'dg', 'dgn', 'ny', 'd', 'u', 'klo',
                       'kalo', 'amp', 'biar', 'bikin', 'bilang',
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih',
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya',
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'p', 'ak',
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah', 'ri', 'dci', 'di', 'iims', 'ge', 
                       'eeehhhh', 'cman', 'pj', 'kyk', 'jrg',
                       'nyahnyoh', 'kya', 'hp', 'jm', 'n', 'ny', 
                       'ama', 'halah', 'entut', 'drun', 'yook', 'dkk', 'a', 'lg',
                       'rd', 'do', 'aq', 'woee', 'q', 'ha', 'brow', 'de',
                       'kq', 'imho', 'hmm', 'ssh', 'aa', 'e', 'tx', 'i', 'iot',
                       'mr', 'co', 'rd', 'dr', 'imho', 'bb', 'eh', 'kl', 'koq', 'ati', 'mw',
                       'lo', 'b', 'pt', 'up', 'aaamiin', 'aama', 'aat', 'zhejiang', 'zzxxx',
                       'zack', 'zarka', 'zimbabwe', 'abang', 'abb', 'abbas', 'abisin',
                       'abrek', 'yudha', 'yuhuu', 'yupss', 'yurioshi', 'yusuf', 'ac',
                       'abrik', 'abu', 'yosua', 'your', 'youtube', 'youtubee', 'yuan', 'mudahhan'])
    txt_stopword = pd.read_csv("stopwords.txt", names=["stopwords"], header=None)
    list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))
    list_stopwords = set(list_stopwords)
    tokens = [word for word in tokens if not word in list_stopwords]
    return tokens

# Fungsi untuk stemming teks
def stemming(Text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    Text = [stemmer.stem(word) for word in Text]
    return Text

# Fungsi untuk menghapus tanda baca
def remove_punct(Text):
    Text = " ".join([char for char in Text if char not in string.punctuation])
    return Text

# Tampilan aplikasi menggunakan Streamlit
st.title("Preprocessing Dataset")
uploaded_file = st.file_uploader("Upload file dataset Excel", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file, usecols=["From-User", "Text", "Label"])
    st.dataframe(df)

    # Remove Netral Label
    df = df[df['Label'] != 'Netral']
    df = df[df['Label'] != 'Netral'].copy()

    # Preprocessing data dengan fungsi cleansing
    df['Cleansing'] = df['Text'].apply(cleansing)

    # Preprocessing data dengan fungsi normalization
    df['Normalization'] = df['Cleansing'].apply(slang_normalization)

    # Tambahkan proses tokenizing setelah normalisasi
    df['Tokenize'] = df['Normalization'].apply(tokenization)

    # Tambahkan proses stopwords setelah tokenizing
    df['Stopwords_Removed'] = df['Tokenize'].apply(remove_stopwords)

    # Tambahkan proses stemming setelah stopwords
    df['Stemming'] = df['Stopwords_Removed'].apply(stemming)

    # Drop duplicate
    df.drop_duplicates(subset='Stemming', keep='first', inplace=True)

    # Reset the index after dropping duplicates
    df = df.reset_index(drop=True)

    # Tambahkan proses penghapusan tanda baca setelah stemming
    df['Comment'] = df['Stemming'].apply(remove_punct)

    # Menampilkan data setelah preprocessing
    st.subheader("Cleansing Result")
    st.dataframe(df[['Cleansing', 'Label']])

    st.subheader("Normalization Result")
    st.dataframe(df[['Normalization', 'Label']])

    st.subheader("Tokenize Result")
    st.dataframe(df[['Tokenize', 'Label']])

    st.subheader("Stopwords Removed Result")
    st.dataframe(df[['Stopwords_Removed', 'Label']])

    st.subheader("Stemming Result")
    st.dataframe(df[['Stemming', 'Label']])

    st.subheader("Comment Result")
    st.dataframe(df[['Comment', 'Label']])

    # Menambahkan tombol download Excel untuk Comment Result
    st.markdown("<h3>Download Comment Result</h3>", unsafe_allow_html=True)
    excel_data = [df[['Comment', 'Label']].columns.tolist()] + df[['Comment', 'Label']].values.tolist()
    wb = Workbook()
    ws = wb.active
    for row in excel_data:
        ws.append(row)
    excel_file = "Preprocessed-Comment-Result.xlsx"
    wb.save(excel_file)
    with open(excel_file, "rb") as file:
        b64 = base64.b64encode(file.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{excel_file}">Download Excel</a>'
        st.markdown(href, unsafe_allow_html=True)