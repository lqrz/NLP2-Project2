import requests
import json
import urllib
import codecs
import time

# IP might get blocked if too many calls...!!!

URL_glosbe = 'https://glosbe.com/gapi/translate?from=eng&dest=deu&format=json&phrase=%s&pretty=true'
URL_trans_yandex = 'https://translate.yandex.net/api/v1.5/tr.json/translate?key=trnsl.1.1.20150522T071527Z.' \
             '0885ef62fc05426d.2547a30105ac655179a793a94ab3ee100cb4994e&lang=en-de&text=%s'


def get_translations_glosbe(word, max_transl=None):
    r = requests.get(URL_glosbe % word)
    data = json.loads(r.text)

    translations = []
    for var in data["tuc"]:
        try:
            translations.append(var["phrase"]["text"])
        except KeyError:
            pass

    return translations[:max_transl]

def get_dict_yandex(word):
    r = requests.get(URL_yandex % urllib.quote(word))
    data = json.loads(r.text)

    translations = []
    syns = []
    for var in data["def"]:
        for tr in var["tr"]:
            translations.append(tr["text"])
            try:
                for syn in tr["syn"]:
                    syns.append(syn["text"])
            except KeyError:
                pass

    return translations + syns

def get_translation_yandex(word):
    r = requests.get(URL_trans_yandex % urllib.quote(word))
    data = json.loads(r.text)


    return data["text"][0] # API returns input word if no better translation found

# print get_translation_yandex("afdadfadfadfafd")

if __name__ == "__main__":

    infile = codecs.open("models/dict/top10ken.txt", "r", "utf-8")

    outfile = codecs.open("models/dict/top10ken_trans_glosbe.txt", "w", "utf-8")

    for line in infile:
        tok = line.strip()
        try:
            outfile.write(tok + "\t" + "\t".join(get_translations_glosbe(tok, max_transl=10)) + "\n")
        except:
            pass

        time.sleep(0.6)

    infile.close()
    outfile.close()


