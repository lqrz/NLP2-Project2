__author__ = 'lqrz'

from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
import sys
from dictionary import get_translations
from word2vec import loadModel
from sklearn.linear_model import Ridge

if __name__ == '__main__':

	#------------- Begin params -------------#
	nWords = 5 # Number of words to train translation matrix
	nTranslations = 1
	alpha = 0.1 # linear regression solver regularization param
	# path = sys.argv[1] # path to .norm file (enlgish)
	# modelPath = sys.argv[2] # path to model file (word2vec)
	modelPath = '/'
	path = '/home/lqrz/Desktop/fold/europarl.en.norm' # path to .norm file (enlgish)
	#------------- End params -------------#

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

		# Get translation from thesaurus api
		translations[word] = get_translations(word)[:nTranslations]

		# Load word2vec trained models (En, De)

		# Get representations for words

		# Instantiate linear regression solver
		solver = Ridge(alpha,solver='lsqr')
		
		# Fit data
		solver.fit(X,y)

		# Store translation matrix
		solver.coef_