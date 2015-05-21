import requests
import json

# IP might get blocked if too many calls...!!!

URL = 'https://glosbe.com/gapi/translate?from=eng&dest=deu&format=json&phrase=%s&pretty=true'



def get_translations(word):
    r = requests.get(URL % word)
    data = json.loads(r.text)

    translations = []
    for var in data["tuc"]:
        try:
            translations.append(var["phrase"]["text"])
        except KeyError:
            pass

    return translations



print get_translations("cat")