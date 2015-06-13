__author__ = 'lqrz'

import pickle
from CorpusReader import CorpusReader
from nltk.corpus import PlaintextCorpusReader
from nltk import FreqDist
import sys
import codecs
from nltk.tokenize import RegexpTokenizer
import re

def getEntries(corpus):
    tableEntries = set()

    for line in corpus.get_raw_corpus():
        entry = line.decode('utf-8').split('|||')[0].strip()
        tableEntries.add(entry)

    return tableEntries


def getTestCorpus(testCorpusPath):
    idx = testCorpusPath.rfind('/') + 1
    folder = testCorpusPath[0:idx]
    filename = testCorpusPath[idx:]
    testCorpus = PlaintextCorpusReader(folder, filename, encoding='utf-8', word_tokenizer=RegexpTokenizer(r'##\w+|\w+|[^\w\s]+'))

    return [w for w in testCorpus.words() if w.isalpha() or re.match(r'^##[a-zA-Z0-9]+$', w)]

def initializeDebug():
    global phraseTablePath1
    global phraseTablePath2
    global testCorpusPath
    global pickled

    phraseTablePath1 = 'table1.gz' # original phrase table
    phraseTablePath2 = 'phrase-table-20.gz' # second phrase table
    testCorpusPath = 'data/test.true.20.de' # test corpus filename
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
    pickled = 'False'
    #----------

    initializeDebug()

    if len(sys.argv) == 5:
        phraseTablePath1 = sys.argv[1] # original phrase table
        phraseTablePath2 = sys.argv[2] # second phrase table
        testCorpusPath = sys.argv[3] # test corpus filename
        pickled = sys.argv[4] # was it previously ran and want to load pickles?
    elif len(sys.argv) > 1:
        print 'Error in params. Usage: 1)Phrase table path, 2) Second phrase table, 3) Test corpus folder, 4) Pickled?'
        exit()

    if pickled == 'False':

        print 'Getting phrase table 1 entries'
        corpusTable1 = CorpusReader(phraseTablePath1)
        table1Entries = getEntries(corpusTable1)

        print 'Getting phrase table 2 entries'
        corpusTable2 = CorpusReader(phraseTablePath2)
        table2Entries = getEntries(corpusTable2)

        print 'Getting test corpus'
        depunctTestCorpus = getTestCorpus(testCorpusPath)

        print 'Constructing test corpus freq dict'
        fd = FreqDist(depunctTestCorpus)

        print 'Pickling data'
        pickle.dump(table1Entries, open(phraseTablePicklePath, 'wb'))
        pickle.dump(table2Entries, open(secondTablePicklePath, 'wb'))
        pickle.dump(fd, open(testCorpusFDPicklePath, 'wb'))

    else:
        print 'Getting pickled phrase table 1 entries'
        table1Entries = pickle.load(open(phraseTablePicklePath, 'rb'))

        print 'Getting pickled phrase table 2 entries'
        table2Entries = pickle.load(open(secondTablePicklePath, 'rb'))

        print 'Getting pickles test corpus freq dict'
        fd = pickle.load(open(testCorpusFDPicklePath, 'rb'))

    print 'Computing stats'

    corpusLen = sum(fd.values())

    # #TODO: test.delete.
    # c2 = PlaintextCorpusReader('data','test.true.de',encoding='utf-8')
    # fd2 = FreqDist([w for w in c2.words() if w.isalpha()])
    # set(fd.keys()) - set(fd2.keys())

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

    print 'Nr of unknown tokens using original table: ', table1OOVTokenNr,  table1OOVTokenPerc, '\n'
    print 'Nr of unknown tokens using original table and second table: ', table2OOVTokenNr,  table2OOVTokenPerc, '\n'
    print 'Total tokens in set: ', corpusLen

    # Pickle OOVs
    pickle.dump(table1OOVs, open('table1OOVs.p', 'wb'))
    pickle.dump(table2OOVs, open('table2OOVs.p', 'wb'))
