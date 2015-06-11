__author__ = 'lqrz'

from CorpusReader import CorpusReader
import sys
from nltk.corpus import PlaintextCorpusReader
import codecs


def getPhraseTableEntries(path):
    tableCorpus = CorpusReader(path)
    tableLines = tableCorpus.get_raw_corpus()

    tableEntries = set()

    sep = '|||'
    for line in tableLines:
        tableEntries.add(line.split(sep)[0].strip())

    return tableEntries

def getTestCorpusWords(path):
    idx = testCorpusPath.rfind('/') + 1
    folder = testCorpusPath[0:idx]
    filename = testCorpusPath[idx:]

    testCorpus = PlaintextCorpusReader(folder, filename)

    depunct = [w for w in testCorpus.words() if w.isalpha()]

    return set (depunct)


if __name__ == '__main__':
    tablePath = 'table1.gz'
    testCorpusPath = 'data/test.true.de'
    outputFilename = 'naturalOOVs.txt'

    if len(sys.argv) == 3:
        tablePath = sys.argv[1]
        testCorpusPath = sys.argv[2]
    elif len(sys.argv) > 1:
        print 'Error in params: 1) Phrase table path 2) Test corpus path'

    tableEntries = getPhraseTableEntries(tablePath)
    testCorpusWords = getTestCorpusWords(testCorpusPath)

    OOVs = testCorpusWords - tableEntries

    fout = codecs.open(outputFilename, 'w', 'utf-8')

    for oov in OOVs:
        fout.write(oov + '\n')

    fout.close()