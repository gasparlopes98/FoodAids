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

def get_labels(df, tokenized_instructions):
    labels = []
    dic_ing={}
    for ing, ti in zip(df["ingredients"], tokenized_instructions):
        l_i = []
        ci = ingredient_parser(ing)
        for i in ci:
            if i not in dic_ing.keys():
                dic_ing[i] = len(dic_ing)
        for token in ti:
            l_i.append(any((c == token.text or c == singularize(token.text) or singularize(c) == token.text) for c in ci))
        labels.append(l_i)
    return labels,dic_ing

def prepare_sequences(texts, max_len, vocab={"<UNK>": 1, "<PAD>": 0}):
    X = [[vocab.get(w.text, vocab["<UNK>"]) for w in s] for s in texts]
    return pad_sequences(maxlen=max_len, sequences=X, padding="post", value=vocab["<PAD>"])

def load_model_RecipeToIngreds(model_path='NLP/model/recipe_ing_model.h5'):
    return tf.saved_model.load(model_path)
    
def train_model_RecipeToIngreds():
    # eval_size =10
    epoch_nr=10
    global vocab
    global dic_ing
    global model
    
    df = pd.read_csv("csv_file/recipes.csv")
    df=df.sample(frac=1).reset_index(drop=True)
    # df = df[eval_size:].reset_index(drop=True)
    # eval_df = df[:eval_size]

    tokenized = [nlp(t) for t in df.recipe.values]
    vocab = {"<UNK>": 1, "<PAD>": 0}
    for txt in tokenized:
        for token in txt:
            if singularize(token.text) not in vocab.keys():
                vocab[singularize(token.text)] = len(vocab)
    labels,dic_ing = get_labels(df, tokenized)
    
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
        
    with open('NLP/save/ingred_vocab.pkl', 'wb') as fp:
        pickle.dump(dic_ing, fp)
    
def get_RecipeToIngreds(input,model):
    
    with open('NLP/save/vocab.pkl', 'rb') as fp:
        vocab = pickle.load(fp)
        
    with open('NLP/save/ingred_vocab.pkl', 'rb') as fp:
        dic_ing = pickle.load(fp)

    input_tokenized=[]
    input_tokenized.append(nlp(input))
    inp_seq = prepare_sequences(input_tokenized, max_len=MAX_LEN, vocab=vocab)
    out_seq = model(inp_seq)
    
    predict = out_seq > 0.01
    ingreds_test = [t.text for t, p in zip(input_tokenized[0], predict[0]) if p]
    set(ingreds_test)
    ing=[]
    for i in ingreds_test:
        if i in dic_ing.keys() and i not in ing:
            ing.append(i)
    return ing
    
if __name__ == "__main__":
    txt = open("test.txt", "r")
    train_model_RecipeToIngreds()
    model = load_model_RecipeToIngreds()
    ingredients = get_RecipeToIngreds(txt.read(),model)
    
    print('Ingredients: ',end="")
    [print(i,end=", ") for i in ingredients]
    print()