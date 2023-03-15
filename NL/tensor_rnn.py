import tensorflow as tf
from tensorflow import keras
import csv
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
nltk.download('stopwords') # install NLTK data to home user directory
from nltk.corpus import stopwords
import unidecode

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@;]')
BAD_SYMBOLS_RE = re.compile('[^a-z 0-9 #+_]')
STOPWORDS = set(stopwords.words('english'))
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
    text = re.sub(r'\n+', ' ', text)

    #text = # delete stopwords from text
    text = STOPWORDS_RE.sub("", text)

    return text


rows = []
dataset = []
with open('receitas/receitas.csv') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)

training_size =3
vocab_size = 10000
embedding_dim = 16

trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
labels=[]
sentences=[]

for receitas in rows:
    with open("receitas/"+receitas[2]) as f:
        recepies = f.readlines()
    label=' '.join(receitas[:2])

    recepies=' '.join(recepies)
    sentences.append(text_prepare(recepies))
    labels.append(label)
    
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]
    
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
max_length = max([len(i) for i in training_sequences])
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_labels_seq = tokenizer.texts_to_sequences(training_labels)
max_length_lab = max([len(i) for i in training_labels_seq])
testing_labels_seq = tokenizer.texts_to_sequences(testing_labels)

training_lab_padded = pad_sequences(training_labels_seq, maxlen=max_length_lab, padding=padding_type, truncating=trunc_type)
testing_lab_padded = pad_sequences(testing_labels_seq, maxlen=max_length_lab, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.SimpleRNN(128),
    tf.keras.layers.Dense(max_length_lab)
    # tf.keras.layers.Dense(24, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

num_epochs = 30
history = model.fit(training_padded, training_lab_padded, epochs=num_epochs, validation_data=(testing_padded, testing_lab_padded), verbose=2)


def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")