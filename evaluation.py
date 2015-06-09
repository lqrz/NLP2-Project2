__author__ = 'lqrz'

import pickle
from CorpusReader import CorpusReader
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import sys

if __name__ == '__main__':

	if len(sys.argv)==4:
		phraseTablePath = sys.argv[1]
		testCorpusFolder = sys.argv[2]
		pickled = sys.argv[3]
	else:
		print 'Usage: 1)Phrase table path, 2) Test corpus folder, 3) pickled'
	if pickled=='False':
		corpus = CorpusReader(phraseTablePath)

		l = set()

		for sentence in corpus.get_raw_corpus():
			l.add(sentence.split('|||')[0].strip())
		pickle.dump(l,open('original_phrasetable.p','wb'))
	
		test_corpus = PlaintextCorpusReader(testCorpusFolder, 'test.true.de')
		fd = FreqDist(test_corpus.words())
		pickle.dump(fd, open('test_freqdict.p','wb'))

	else:
		print 'Getting pickled table set'
		l = pickle.load(open('original_phrasetable.p', 'rb'))
		fd = pickle.load(open('test_freqdict.p','rb'))
	print 'Getting test corpus'
	# test_corpus = PlaintextCorpusReader('corpus/testing/','test.true.de')

	oovs = len(set(fd.keys()) - l)
	oovs_perc = oovs / len(fd.keys())

	print 'oovs: ', oovs, oovs_perc, '\n'
	print 'Total words in set: ', len(fd.keys())
