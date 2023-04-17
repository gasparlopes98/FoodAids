# import os
# os.chdir('/home/golopes/mestrado/projetos/FoodAid/')
import unidecode
import pandas as pd
import embeddings as emb

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from ingredient_parser import ingredient_parser

csv_file = "csv_file/recipes.csv"

def get_and_sort_corpus(data):
    """
    Get corpus with the documents sorted in alphabetical order
    """
    corpus_sorted = []
    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

def ingredient_parser_final(ingredient):
    """
    neaten the ingredients being outputted
    """
    if isinstance(ingredient, list):
        ingredients = ingredient
    else:
        ingredients = ingredient.split()

    ingredients = ",".join(ingredients)
    ingredients = unidecode.unidecode(ingredients)
    return ingredients

def get_recommendations(N, scores,learning_param):
    """
    Top-N recomendations order by score
    """
    
    # load in recipe dataset
    df_recipes = pd.read_csv(csv_file)
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    learning_param.append("recipe")
    recommendation = pd.DataFrame(columns=learning_param)
    learning_param.append("score")

    for i in top:
        # recommendation.at[i, "title"] = unidecode.unidecode(df_recipes["title"][i])
        for param in learning_param[0:-1]:
            recommendation.at[i, param] = unidecode.unidecode(
                df_recipes[param][i]
            )
        recommendation.at[i, "score"] = f"{scores[i]}"
    return recommendation
def load_word2vec():
    model_path = "NLP/model/model_key_recipe.model"
    
    model = Word2Vec.load(model_path)
    model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")
    return model
    
def get_recipes_keywords(model,input, N=1, mean=False):
    # load in word2vec model
    learning_param = ["title","ingredients","region"]
    
    # load in data
    data = pd.read_csv(csv_file)
    
    # parse parameters
    parameters=pd.DataFrame()
    for i in learning_param:
        if parameters.empty:
            parameters = data[i]
        else:
            parameters = parameters + ',' + data[i]
    data["parsed"] = parameters.apply(ingredient_parser)
    
    # create corpus
    corpus = get_and_sort_corpus(data)

    if mean:
        # get average embdeddings for each document
        vec_tr = emb.MeanEmbeddingVectorizer(model)
    else:
        # use TF-IDF as weights for each word embedding
        vec_tr = emb.TfidfEmbeddingVectorizer(model)
        
    vec_tr.fit(corpus)
    doc_vec = vec_tr.transform(corpus)
    doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
    # assert len(doc_vec) == len(corpus)

    # parse ingredient list
    input = ingredient_parser(input,keyword=True)
    # get embeddings for ingredient doc
    input_embedding = vec_tr.transform([input])[0].reshape(1, -1)

    # get cosine similarity between input embedding and all the document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    # Filter top N recommendations
    recommendations = get_recommendations(N, scores,learning_param)
    return recommendations

if __name__ == "__main__":
    input = "give me recipe of francesinha"
    n_sugestions=3
    
    rec = get_recipes_keywords(input,n_sugestions)
    print("For this key words: ", input)
    print(rec)