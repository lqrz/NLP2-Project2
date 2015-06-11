__author__ = 'lqrz'
import sys
from itertools import combinations
import pickle
from gensim import gensim
import numpy as np
import codecs
from os import walk
import re
import gzip
from nltk.corpus import PlaintextCorpusReader

if __name__ == '__main__':

    nCandidates = 10 # number of possible translations to retrieve from embedding space
    processAll = False

    modelsPath = '/home/wechsler/NLP2-Project2/models/truecase/'
    directPath = modelsPath + 'de-en/yandex/'
    inversePath = modelsPath + 'en-de/'

    if len(sys.argv) == 5:
        modelsPath = sys.argv[1]
        oovFilename = sys.argv[2]
	directPath = sys.argv[3]
	inversePath = sys.argv[4]
    elif len(sys.argv) > 1 :
        print 'Error! missing parameters'

    dimensions = [200,400,800]

    # oovFilenames = ['oovs24.5998608212.p'] # TODO: Add pickle filenames

    oovFilenames = []
    if oovFilename == 'False':
        print 'Processing all wordsToReplace files in current directory. \n'
        for (_, _, filenames) in walk('./'):
            files = [re.match(r'^(wordsToReplace).*',fn).group() for fn in filenames if re.match(r'^(wordsToReplace).*',fn)]
            oovFilenames.extend(files)
    else:
        print 'Processing file: ' + oovFilename + '\n'
        oovFilenames.append(oovFilename)


    tmFilenames = set()
    modelFilenames = set()

    for d in dimensions:
        d1 = str(d)
        tmFilenames.add(('mono_'+d1+'_de.bin', 'mono_'+d1+'_en.bin', 'tm_'+ d1 +'_'+ d1 +'.p', 'tm_'+ d1 +'_'+ d1 +'.p'))

    for p in combinations(dimensions,2):
        de = str(p[0])
        en = str(p[1])
        tmFilenames.add(('mono_'+de+'_de.bin', 'mono_'+en+'_en.bin', 'tm_'+de+'_'+en+'.p', 'tm_'+en+'_'+de+'.p'))


    tmFilenames = [('mono_800_de.bin', 'mono_400_en.bin', 'tm_800_400.p', 'tm_400_800.p')]
#    directPath = 'models/'
#    inversePath = 'models/'
#    modelsPath = 'models/'

    for i,t in enumerate(tmFilenames):
        deModel = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[0],binary=True)
        enModel = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[1],binary=True)
        directMatrix = pickle.load(open(directPath + t[2]))
        inverseMatrix = pickle.load(open(inversePath + t[3]))

    noTranslationWords = set()

    # load OOVs
    for f in oovFilenames:
#        outputPhraseTablePath = 'secondPhraseTable' + f + '.gz'
        outputPhraseTablePath = 'secondPhraseTable' + f
        outputLogFilePath = 'secondTable' + f + '.log'

        foutPhraseTable = codecs.open(outputPhraseTablePath, "w", "utf-8")
#        foutPhraseTable = gzip.open(outputPhraseTablePath, "wb")
        foutLog = codecs.open(outputLogFilePath, "w", "utf-8")

        # oovs = pickle.load(open(f,'rb'))

        corpus = PlaintextCorpusReader('./', f, "utf-8")
        oovs = corpus.words()

        totalOOV = len(oovs)
        foutLog.write('Processing file ' + f + '\n')
        foutLog.write('Total number of unknown words: ' + str(totalOOV) + '\n')
        for oov in oovs:
            try:
                # get representations
                deRep = deModel[oov]
                calculatedRep = np.dot(directMatrix, deRep)
                candidates = enModel.most_similar(positive=[calculatedRep], topn=nCandidates)

                totalProb = 0
                for wCand, wCos in candidates:
                    totalProb += wCos
                    try:
                        inverseRep = enModel[wCand]
                        inverseCalculatedRep = calculatedRep = np.dot(inverseMatrix, inverseRep)
                        inverseCandidates = deModel.most_similar(positive=[inverseCalculatedRep], topn=nCandidates)
                        tot = 0
                        if any(oov in trans for trans in inverseCandidates):
                            idx = next(i for i, (trans, _) in enumerate(inverseCandidates) if trans == oov)
                            tot += sum([cos for _,cos in inverseCandidates])
                            inverseProb = inverseCandidates[idx][1] / float(tot)
                        else:
                            inverseProb = 0
                    except:
                        inverseProb = 0

                for wCand, wCos in candidates:
                    convOOV = '##' + oov
                    sep = '|||'
                    directProb = str(wCos / float(totalProb))
                    inverseProb = str(inverseProb)
                    line = ' '.join([convOOV,sep,wCand,sep,inverseProb, inverseProb,directProb,directProb,sep,sep,sep,sep,'\n'])
                    foutPhraseTable.write(line)


                # construct probabilities
            except:
                noTranslationWords.add(oov)
                continue

        foutLog.write('Total words translated: ' + str(totalOOV - len(noTranslationWords)) + '\n')
        for w in set(oovs) - set(noTranslationWords):
            foutLog.write('Translated word: ' + w + '\n')

        for w in noTranslationWords:
            foutLog.write('Untranslated word: ' + w + '\n')

        foutLog.close()
        # save second table
        foutPhraseTable.close()
