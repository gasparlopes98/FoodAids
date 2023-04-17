from rest_framework.decorators import api_view
from django.http import HttpResponse
import json


@api_view(['GET', 'POST', 'DELETE'])
def index(request):
    return HttpResponse(json.dumps({"text":"hello"}))
