import os
import tensorflow as tf
from keras_preprocessing.sequence import pad_sequences
import keras
from tensorflow.keras import layers
import spacy
nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger', 'ner'])
from ingredient_parser import ingredient_parser
from pattern.text.en import singularize
import numpy as np
import pandas as pd
import pickle
MAX_LEN=400

class NLModel(keras.Model):
    def __init__(self, vocab_len, output_dim,name=None):
        super().__init__(name=name)
        self.vocab_len = vocab_len
        self.output_dim = output_dim
        self.model = tf.keras.Sequential([
            layers.Embedding(input_dim=vocab_len, mask_zero=True, output_dim=output_dim),
            layers.SpatialDropout1D(0.2),
            layers.Bidirectional(layers.LSTM(units=64, return_sequences=True)),
            layers.SpatialDropout1D(0.2),
            layers.Bidirectional(layers.LSTM(units=64, return_sequences=True)),
            layers.TimeDistributed(layers.Dense(1, activation='sigmoid'))
        ])
        
    def call(self, inputs):
        x = self.model(inputs)
        return x

def clear_input(_):
    return ''

def get_labels(df, tokenized_instructions):
    labels = []
    for ing, ti in zip(df["ingredients"], tokenized_instructions):
        l_i = []
        ci = ingredient_parser(ing)
        for token in ti:
            l_i.append(any((c == token.text or c == singularize(token.text) or singularize(c) == token.text) for c in ci))
        labels.append(l_i)
    return labels

def prepare_sequences(texts, max_len, vocab={"<UNK>": 1, "<PAD>": 0}):
    X = [[vocab.get(w.text, vocab["<UNK>"]) for w in s] for s in texts]
    return pad_sequences(maxlen=max_len, sequences=X, padding="post", value=vocab["<PAD>"])

def load_model_RecipeToIngreds(model_path='NLP/model/recipe_ing_model.h5'):
    return tf.saved_model.load(model_path)
    
def train_model_RecipeToIngreds():
    epoch_nr=10
    
    df = pd.read_csv("csv_file/new_recipes.csv")
    df=df.sample(frac=1).reset_index(drop=True)

    tokenized=[]
    for t in df.recipe.values:
        try:
            tokenized.append(nlp(t))
        except:
            pass
        
    vocab = {"<UNK>": 1, "<PAD>": 0}
    for txt in tokenized:
        for token in txt:
            if singularize(token.text) not in vocab.keys():
                vocab[singularize(token.text)] = len(vocab)
    labels = get_labels(df, tokenized)
    
    X_seq = prepare_sequences(tokenized, max_len=MAX_LEN, vocab=vocab)

    y_seq = []
    for l in labels:
        y_i = []
        for i in range(MAX_LEN):
            try:
                y_i.append(float(l[i]))
            except:
                y_i.append(0.0)
        y_seq.append(np.array(y_i))
    y_seq = np.array(y_seq)
    y_seq = y_seq.reshape(y_seq.shape[0], y_seq.shape[1], 1)
    
    model=NLModel(len(vocab),50)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='binary_crossentropy',
                metrics=['accuracy'])

    model.fit(X_seq, y_seq, epochs=epoch_nr, batch_size=256, validation_split=0.1)
    print('Model Trained!')
    
    tf.saved_model.save(model, 'NLP/model/recipe_ing_model.h5')

    with open('NLP/save/vocab.pkl', 'wb') as fp:
        pickle.dump(vocab, fp)
    
def get_RecipeToIngreds(txt,model):
    
    with open('NLP/save/vocab.pkl', 'rb') as fp:
        vocab = pickle.load(fp)

    input_tokenized=[]
    input_tokenized.append(nlp(txt))
    inp_seq = prepare_sequences(input_tokenized, max_len=MAX_LEN, vocab=vocab)
    out_seq = model(inp_seq)
    
    predict = out_seq > 0.1
    ingreds_test = [t.text for t, p in zip(input_tokenized[0], predict[0]) if p]
    set(ingreds_test)
    ing=[]
    for i in ingreds_test:
        if singularize(i) not in ing:
            ing.append(i)
            
    ing=str(','.join(ing))
    
    clear_input
    title = input('Give Recipe Title: ')
    df = pd.read_csv("csv_file/new_recipes.csv")
    df.loc[len(df)]={'title':title,'ingredients':ing,'recipe':txt}
    df.to_csv('csv_file/new_recipes.csv', index=False)
    
    return ing
    
if __name__ == "__main__":
    txt = open("test2.txt", "r")
    # train_model_RecipeToIngreds()
    model = load_model_RecipeToIngreds()
    ingredients = get_RecipeToIngreds(txt.read(),model)
    
    print('Ingredients: '+ingredients)