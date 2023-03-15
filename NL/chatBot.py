import spacy
import random
import re
import nltk
nltk.download('stopwords') # install NLTK data to home user directory
from nltk.corpus import stopwords
import unidecode

file_name = "receitas_bacalhau_bras/1_bacalhau_bras.txt"
f = open(file_name, 'r')
file_contents = f.read()

nlp = spacy.load("pt_core_news_sm")

def clear_input(_):
    return ''

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
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
    response = ["Não percebi... ",
            "?...",
            "Acho que não sou capaz de responder a isso.",
            "O que é que isso significa?"][random.randrange(4)]
    return response

def check_all_messages(message):
    highest_prob_list = {}

    # Simplifies response creation / adds it to the dict
    def response(bot_response, list_of_words, single_response=False, required_words=[]):
        nonlocal highest_prob_list
        highest_prob_list[bot_response] = message_probability(message, list_of_words, single_response, required_words)

    # Responses -------------------------------------------------------------------------------------------------------
    response('Olá!', ['ola', 'hey'], single_response=True)
    response('Adeus!', ['adeus', 'ate', 'breve', 'xau'], single_response=True)
    response('Estou bem e tu?', ['como', 'estas', 'tudo', 'bem'], single_response=True)
    response('De nada!', ['obrigado'], single_response=True)
    response('Ok, aqui vai: \n' + file_contents, ['da', 'receita', 'bacalhau'], required_words=['receita', 'bacalhau'])
    
    best_match = max(highest_prob_list, key=highest_prob_list.get)
    # print(highest_prob_list)
    # print(f'Best match = {best_match} | Score: {highest_prob_list[best_match]}')

    return unknown() if highest_prob_list[best_match] < 1 else best_match

# Used to get the response
def get_response(user_input):
    text = text_prepare(user_input)
    print(text)
    doc = nlp(text)
    response = check_all_messages([token.text for token in doc])
    return response


clear_input
while True:
    inp=input('Tu: ')
    print('Bot: ' + get_response(inp))
        