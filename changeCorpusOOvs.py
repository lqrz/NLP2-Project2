__author__ = 'lqrz'

import sys
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import random
import codecs
# import gensim
from gensim import gensim

if __name__ == '__main__':

    # nWords = 100 # Consider n most frequent words to eliminate
    minFreq = 10
    maxFreq = 50

    testCorpusPath = 'data/test.true.de'
    naturalOOVsFilename = 'naturalOOVs.txt'
    modelPath = 'models/mono_800_de.bin'
    if len(sys.argv) == 4:
        testCorpusPath = sys.argv[1]
        naturalOOVsFilename = sys.argv[2]
        modelPath = sys.argv[3]
    elif len(sys.argv) > 1:
        print 'Error in params: 1) Test corpus path 2) Natural OOVs filename 3) Word2vec model path'
        exit()

    idx = testCorpusPath.rfind('/') + 1
    folder = testCorpusPath[0:idx]
    filename = testCorpusPath[idx:]

    print 'Retrieving file entries'
    corpus = PlaintextCorpusReader(folder, filename, encoding='utf-8')
    naturalCorpus = PlaintextCorpusReader('./',naturalOOVsFilename, encoding='utf-8')

    print 'Retrieving word2vec model'
    model = gensim.models.Word2Vec.load_word2vec_format(modelPath,binary=True)

    naturalOOVs = set(naturalCorpus.words())

    depunct = [w for w in corpus.words() if w.isalpha()]

    fd = FreqDist(depunct)

    totalLen = sum(fd.values())

    mostFrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)

    filteredWords = [(w,f) for w,f in mostFrequentWords if f>=minFreq and f<=maxFreq ]

    # base = sum([f for w,f in fd.items() if w in naturalOOVs]) / float(totalLen)
    base = sum([fd[w] for w in naturalOOVs]) / float(totalLen) # OOV percentage wrt tokens

    percs = [20, 50, 70]

    for p in percs:
        print 'Processing perc: ', p
        tot = base
        candidates = set()
        while tot < p:
            candidate = random.choice(filteredWords)

            # check we are adding a new unknown to the natural set
            if candidate[0] in naturalOOVs:
                continue

            # check we can get a translation for it
            try:
                model[candidate[0]]
            except KeyError:
                continue

            tot += candidate[1]*100 / float(totalLen)
            candidates.add(candidate[0]) # add word to remove from phrase table or corpus

        fout = codecs.open('wordsToReplace'+str(tot)+'.txt', 'w', 'utf-8')

        # write naturalOOVs
        for w in naturalOOVs:
            fout.write(w+'\n')

        # write artificialOOVs
        for w in candidates:
            fout.write(w+'\n')

        fout.close()
