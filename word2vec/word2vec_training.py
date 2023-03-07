import pandas as pd 
from gensim.models import Word2Vec
from ingredient_parser import ingredient_parser

# get corpus with the documents sorted in alphabetical order
def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.parsed.values:
        doc.sort()
        corpus_sorted.append(doc)
    return corpus_sorted

class Model:
    def __init__(self, sg, workers,window,min_count):
        self.sg = sg
        self.workers = workers
        self.window = window
        self.min_count = min_count
        
    def training(self,file,output_file,lerning_param):
        data = pd.read_csv(file)
        parameters=pd.DataFrame()
        for i in lerning_param:
            if parameters.empty:
                parameters = data[i]
            else:
                parameters = parameters + ',' + data[i]
            # parameters=parameters +','+ data[i]
        # data["parsed"] = parameters.apply(
        #     lambda x: ingredient_parser(x)
        # )
        
        data['parsed'] = parameters.apply(ingredient_parser)
        data.head()

        corpus = get_and_sort_corpus(data)
        print(f"Length of corpus: {len(corpus)}")

        model_cbow = Word2Vec(corpus, sg=self.sg, workers=self.workers, window=self.window, min_count=self.min_count, vector_size=100)

        #Summarize the loaded model
        print(model_cbow)

        #Summarize vocabulary
        words = list(model_cbow.wv.index_to_key)
        words.sort()

        #Acess vector for one word
        # print(model_cbow.wv['chicken stock'])

        # most similar
        # model_cbow.wv.most_similar(u'sugar')
        # model_cbow.wv.similarity('cauliflower', 'cauliflower just larger than potato')
        model_cbow.save(output_file)
        

if __name__ == "__main__":
    modelTraining=Model(sg = 0, # CBOW: build a language model that correctly predicts the center word given the context words in which the center word appears
        workers = 8, # number of CPUs
        window = 6, # window size: average length of each document 
        min_count = 1 # unique ingredients are important to decide recipes 
        )
    
    input_file = 'word2vec/recipes.csv'
    output_file = 'word2vec/models/model_ingredients_region.model'
    lerning_param = ["ingredients","region"]
    
    modelTraining.training(input_file,output_file,lerning_param)