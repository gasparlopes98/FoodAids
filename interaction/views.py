from rest_framework.decorators import api_view
from django.http import HttpResponse
import json
from NLP.chatBot import get_recipes_keywords_wrapper
from NLP.chatBot import get_response


@api_view(['GET', 'POST', 'DELETE'])
def index(request):
    user_question = request.data["prompt"]
    answer = get_response(user_question)
    if "Recipe" in answer:
        recipe=get_recipes_keywords_wrapper(user_question)
        recipe_info = "Title: " + ' '.join(recipe.title.values) + "\n"
        recipe_info += "Ingredients:" + "\n" + '\n'.join(recipe.ingredients.values)
        recipe_info += "Preparation:" + '\n' + '\n'.join(recipe.recipe.values)
        answer = recipe_info

    return HttpResponse(json.dumps({"text":answer}))
