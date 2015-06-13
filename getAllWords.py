__author__ = 'lqrz'

from nltk.corpus import PlaintextCorpusReader
import sys
import codecs

if __name__ == "__main__":
    foutFilename = 'allWords.txt'
    testCorpusPath = 'data/test.true.de'
    if len(sys.argv) == 3:
        testCorpusPath = sys.argv[1]
        foutFilename = sys.argv[2]
    elif len(sys.argv) > 1:
        print 'Error in params: 1) Test corpus path 2) Output filename'
        exit()

    idx = testCorpusPath.rfind('/') + 1
    folder = testCorpusPath[0:idx]
    filename = testCorpusPath[idx:]

    corpus = PlaintextCorpusReader(folder, filename, encoding='utf-8')

    depunct = set([w for w in corpus.words() if w.isalpha()])

    fout = codecs.open(foutFilename, 'w', encoding='utf-8')

    for w in depunct:
        fout.write(w + '\n')

    fout.close()