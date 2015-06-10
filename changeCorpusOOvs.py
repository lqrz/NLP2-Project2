__author__ = 'lqrz'

import sys
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import random
import codecs

if __name__ == '__main__':

    nWords = 100 # Consider n most frequent words to eliminate

    testCorpusPath = 'data/test.true.de'
    if len(sys.argv) == 2:
	testCorpusPath = sys.argv[1]

    idx = testCorpusPath.rfind('/') + 1
    folder = testCorpusPath[0:idx]
    filename = testCorpusPath[idx:]

    corpus = PlaintextCorpusReader(folder, filename)

    depunct = [w for w in corpus.words() if w.isalpha()]

    fd = FreqDist(depunct)

    totalLen = sum(fd.values())

    mostFrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)[:nWords]

    percs = [20,50,70]
    tot = 0
    candidates = set()
    for p in percs:
        while tot < p:
            candidate = random.choice(mostFrequentWords)
            tot += candidate[1]*100 / float(totalLen)
            candidates.add(candidate[0]) # add word to remove from phrase table or corpus
        fout = codecs.open('wordsToReplace'+str(tot)+'.txt','w','utf-8')
        for w in candidates:
            fout.write(w+'\n')
        fout.close()
