__author__ = 'lqrz'

from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
import sys
from dictionary import get_translations
from word2vec import loadModel

if __name__ == '__main__':

	nWords = 5 # Number of words to train translation matrix
	nTranslations = 1

	# path = sys.argv[1] # path to .norm file (enlgish)
	# modelPath = sys.argv[2] # path to model file (word2vec)
	modelPath = '/'
	path = '/home/lqrz/Desktop/fold/europarl.en.norm' # path to .norm file (enlgish)
	idx = path.rfind('/')+1
	folder = path[0:idx]
	filename = path[idx:]

	wordlist = PlaintextCorpusReader(folder,filename)

	# Construct frequency dictionary
	fd = FreqDist(wordlist.words())

	# Get translations from api En -> De
	translations = dict()

	mostFrequentWords = sorted(fd.items(),key=lambda x:x[1],reverse=True)[:nWords]

	model = loadModel()

	for word,_ in mostFrequentWords:
		translations[word] = get_translations(word)[:nTranslations]
		model.