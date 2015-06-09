__author__ = 'lqrz'

import pickle
from CorpusReader import CorpusReader
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist


if __name__ == '__main__':
	pickled = True

	if not pickled:
		corpus = CorpusReader('corpus/truecased/5k/train/model/phrase-table.gz')

		l = set()

		for sentence in corpus.get_raw_corpus():
			l.add(sentence.split('|||')[0])
		pickle.dump(l,open('original_phrasetable.p','wb'))
	else:
		print 'Getting pickled table set'
		l = pickle.load(open('original_phrasetable.p', 'rb'))

	print 'Getting test corpus'
	# test_corpus = PlaintextCorpusReader('corpus/testing/','test.true.de')
	test_corpus = PlaintextCorpusReader('corpus/testing/', 'test.true.de')
	fd = FreqDist(test_corpus.words())

	coverage = len(fd.keys() - l)

	print 'Word coverage', coverage