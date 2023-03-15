import spacy
import pathlib
import pandas as pd
import re
import nltk
nltk.download('stopwords') # install NLTK data to home user directory
from nltk.corpus import stopwords
import unidecode
import csv
import torchtext
from torchtext.data import get_tokenizer
tokenizer = get_tokenizer("basic_english")

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@;],')
BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')
STOPWORDS = set(stopwords.words('portuguese'))
STOPWORDS_RE = re.compile(r"\b(" + "|".join(STOPWORDS) + ")\\W")

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    
    #text = # lowercase text
    text = unidecode.unidecode(text).lower()
    #text = # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    #text = # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub(" ", text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    #text = # delete stopwords from text
    text = STOPWORDS_RE.sub("", text)

    return text

nlp = spacy.load("pt_core_news_sm")

rows = []
dataset = []
with open('receitas/receitas.csv') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)
# print(header)
# print(rows)

for receitas in rows:
    with open(receitas[2]) as f:
        lines = f.readlines()
    tokens = tokenizer(lines)
    # print(' '.join(receitas[:2]))
    # text=text_prepare(' '.join(receitas[:2]))
    # print(text)
    # doc = nlp(text)

    # features = []
    # for token in doc:
    #     features.append({'token' : token.text, 'pos' : token.pos_})
    
    # dataset.append((receitas[2],features))

    # fdf = pd.DataFrame(features)
    # fdf.head(len(fdf))

# print(dataset)

with open('receitas/dataset.csv', 'w') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerows(dataset)