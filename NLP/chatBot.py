from sys import path
path.append('image_rec')
import predict_dish as predict

import spacy
import random
import re
from nltk.corpus import stopwords
import unidecode
from word2vec import load_word2vec,get_recipes_keywords
from recipe_to_ingredients import load_model_RecipeToIngreds,get_RecipeToIngreds

# Loading models
model_w2v=load_word2vec()
model_recie_ing = load_model_RecipeToIngreds()
model_img=predict.read_model()

nlp = spacy.load("en_core_web_sm")

def clear_input(_):
    return ''

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
STOPWORDS_RE = re.compile(r"\b(" + "|".join(STOPWORDS) + ")\\W")

def text_prepare(text):
    #text = # lowercase text
    text = unidecode.unidecode(text).lower()
    #text = # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = REPLACE_BY_SPACE_RE.sub(" ", text)
    #text = # delete symbols which are in BAD_SYMBOLS_RE from text
    text = BAD_SYMBOLS_RE.sub("", text)
    # remove multiple spaces
    text = re.sub(r'\s+', ' ', text)

    #text = # delete stopwords from text
    text = STOPWORDS_RE.sub("", text)

    return text

def message_probability(user_message, recognised_words, single_response=False, required_words=[]):
    message_certainty = 0
    has_required_words = True

    # Counts how many words are present in each predefined message
    for word in user_message:
        if word in recognised_words:
            message_certainty += 1

    # Calculates the percent of recognised words in a user message
    percentage = float(message_certainty) / float(len(recognised_words))

    # Checks that the required words are in the string
    for word in required_words:
        if word not in user_message:
            has_required_words = False
            break

    # Must either have the required words, or be a single response
    if has_required_words or single_response:
        return int(percentage * 100)
    else:
        return 0

def unknown():
    response = ["I didin't understand can you repeat... ",
                "?...",
                "I don't think I know how to respond to that.",
                "What do you mean?"][random.randrange(4)]
    return response

def check_all_messages(message):
    highest_prob_list = {}

    # Simplifies response creation / adds it to the dict
    def response(bot_response, list_of_words, single_response=False, required_words=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, single_response, required_words)

    # Responses -------------------------------------------------------------------------------------------------------
    response('Hello! How can I help you?', ['hi', 'hey','hello'], single_response=True)
    response('Goodbye!', ['See you', 'goodbye','bye'], single_response=True)
    response('I am ok. And you?', ['how', 'are', 'you'], single_response=False)
    response('You are Welcome', ['thank you','thanks'], single_response=True)
    response('Recipe: ', ['give','recipe'], single_response=False)
    response('Dish in image is: ', ['.jpg'], single_response=True)
    response('The ingredients are: ', ['give','ingredients','.txt'], single_response=False)
    
    best_match = max(highest_prob_list, key=highest_prob_list.get)

    return unknown() if highest_prob_list[best_match] < 1 else best_match


def extract_recipe_name(user_input):
    doc = nlp(user_input)
    # extract noun chunks that are likely to be dish names
    recipe_names = [chunk.text for chunk in doc.noun_chunks if "recipe" in chunk.root.head.text.lower()]
    return recipe_names

# Used to get the response
def get_response(user_input):
    text = text_prepare(user_input)
    doc = nlp(text)
    recipe_names = extract_recipe_name(text) # extract dish name from user input
    if recipe_names: # if dish name is found
        response = generate_response(recipe_names) # generate response using dish name
    else:
        response = check_all_messages([token.text for token in doc]) # generate response without dish name
    return response

def get_recipes_keywords_wrapper(input):
    return get_recipes_keywords(model_w2v,input)

def generate_response(recipe_names):
    output = "The following recipes exist for the dish name you mentioned: " + ', '.join(recipe_names)
    return output

chat=['','','']
def insert_chat(word):
    chat.pop(0)
    chat.append(word)
    return chat

clear_input
if __name__ == "__chatBot__":
    while True:
        inp=input('user: ')
        insert_chat(inp)
        resp=get_response(inp)
        
        if resp == 'Goodbye!':
            print('FoodAids: ' + resp)
            break;

        elif ".jpg" in inp:
            x=inp.split(' ')
            try:
                response = predict.predict_image(model_img,x[-1])
                print('FoodAids: Dish in image is: ' + ' '.join(response.split('_')))
                insert_chat(' '.join(response.split('_')))
            except:
                print('FoodAids: Did not find path!')          

        elif resp == 'Recipe: ':
            print('FoodAids: ' + resp)
            for i in range(2,-1,-1):
                print(chat[i])
                recipe=get_recipes_keywords(model_w2v,chat[i])
                if float(recipe.score.values[0]) > 0.20: 
                    print('Title: ',end="")
                    print(' '.join(recipe.title.values))
                    print('Ingredients: ',end="")
                    [print(i) for i in recipe.ingredients.values]
                    print('Preparation: ',end="")
                    print(' '.join(recipe.recipe.values))
                    break
            
        elif resp == 'The ingredients are: ':
            x=inp.split(' ')
            
            try:
                txt = open(x[-1], "r")
                ingredients = get_RecipeToIngreds(txt.read(),model_recie_ing)
                print('FoodAids: ' + resp, end="")
                print(ingredients)
            except:
                print('FoodAids: Did not find path!')
        else:
            print('FoodAids: ' + resp)