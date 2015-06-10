__author__ = 'lqrz'

import pickle
from CorpusReader import CorpusReader
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import sys

def getEntries(corpus):
    tableEntries = set()

    for sentence in corpus.get_raw_corpus():
        tableEntries.add(sentence.split('|||')[0].strip())

    return tableEntries


def getTestCorpus(testCorpusPath):
    idx = testCorpusPath.rfind('/') + 1
    folder = testCorpusPath[0:idx]
    filename = testCorpusPath[idx:]
    testCorpus = PlaintextCorpusReader(folder, filename)

    return [w for w in testCorpus.words() if w.isalpha()]

def initializeDebug():
    global phraseTablePath1
    global phraseTablePath2
    global testCorpusPath
    global pickled

    phraseTablePath1 = 'table1.gz' # original phrase table
    phraseTablePath2 = 'table2.gz' # second phrase table
    testCorpusPath = 'data/test.true.de' # test corpus filename
    pickled = 'False' # was it previously ran and want to load pickles?

    return True

if __name__ == '__main__':
    #---------- Params
    testCorpusFDPicklePath = 'testCorpusFreqDict.p'
    phraseTablePicklePath = 'originalPhraseTableEntries.p'
    secondTablePicklePath ='secondTableEntries.p'

    phraseTablePath1 = None
    phraseTablePath2 = None
    testCorpusPath = None
    pickled = None
    #----------

    initializeDebug()

    if len(sys.argv) == 5:
        phraseTablePath1 = sys.argv[1] # original phrase table
        phraseTablePath2 = sys.argv[2] # second phrase table
        testCorpusPath = sys.argv[3] # test corpus filename
        pickled = sys.argv[4] # was it previously ran and want to load pickles?
    elif len(sys.argv) > 1:
        print 'Usage: 1)Phrase table path, 2) Second phrase table, 3) Test corpus folder, 3) Pickled?'

    if pickled=='False':
        corpusTable1 = CorpusReader(phraseTablePath1)
        corpusTable2 = CorpusReader(phraseTablePath2)

        table1Entries = getEntries(corpusTable1)
        table2Entries = getEntries(corpusTable2)

        pickle.dump(table1Entries, open(phraseTablePicklePath, 'wb'))
        pickle.dump(table2Entries, open(secondTablePicklePath, 'wb'))

        depunctTestCorpus = getTestCorpus(testCorpusPath)

        fd = FreqDist(depunctTestCorpus)

        pickle.dump(fd, open(testCorpusFDPicklePath, 'wb'))

    else:
        print 'Getting pickled table set'
        table1Entries = pickle.load(open(phraseTablePicklePath, 'rb'))
        table2Entries = pickle.load(open(secondTablePicklePath, 'rb'))
        fd = pickle.load(open(testCorpusFDPicklePath, 'rb'))

    print 'Getting test corpus'
    # test_corpus = PlaintextCorpusReader('corpus/testing/','test.true.de')

    corpusLen = sum(fd.values())

    allEntries = table1Entries.union(table2Entries)

    table1OOVs = set(fd.keys()) - table1Entries
    table2OOVs = set(fd.keys()) - allEntries

    # Word type stats
    totalWordTypes = len(fd.keys())

    table1OOVNr = len(table1OOVs)
    table2OOVNr = len(table2OOVs)
    table1OOVsPerc = float(table1OOVNr)*100 / totalWordTypes # percentage of word types
    table2OOVsPerc = float(table2OOVNr)*100 / totalWordTypes # percentage of word types

    # Token stats
    table1OOVTokenNr = 0
    for w in table1OOVs:
        table1OOVTokenNr += fd[w]

    table2OOVTokenNr = 0
    for w in table2OOVs:
        table2OOVTokenNr += fd[w]

    table1OOVTokenPerc = float(table1OOVTokenNr)*100 / corpusLen
    table2OOVTokenPerc = float(table2OOVTokenNr)*100 / corpusLen

    # Print stats
    print 'Nr of unknown word types using original table: ', table1OOVNr, table1OOVsPerc, '\n'
    print 'Nr of unknown word types using original table and second table: ', table2OOVNr, table2OOVsPerc, '\n'
    print 'Total word types in set: ', totalWordTypes

    print 'Nr of unknown tokens using original table: ',table1OOVTokenNr,  table1OOVTokenPerc,'\n'
    print 'Nr of unknown tokens using original table and second table: ',table2OOVTokenNr,  table2OOVTokenPerc,'\n'
    print 'Total tokens in set: ', corpusLen

    # Pickle OOVs
    pickle.dump(table1OOVs,open('table1OOVs.p', 'wb'))
    pickle.dump(table2OOVs,open('table1OOVs.p', 'wb'))
