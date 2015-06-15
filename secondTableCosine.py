__author__ = 'lqrz'
import sys
# from itertools import combinations
import pickle
# from gensim import gensim
import gensim
import numpy as np
import codecs
from os import walk
import re
from nltk.corpus import PlaintextCorpusReader

if __name__ == '__main__':

    nDirectCandidates = 10 # number of possible translations to retrieve from embedding space
    nInverseCandidates = 1
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

    # dimensions = [200,400,800]

    oovFilename = 'wordsToReplace15.txt' # TODO: For debug. Comment.

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

    # for d in dimensions:
    #     d1 = str(d)
    #     tmFilenames.add(('mono_'+d1+'_de.bin', 'mono_'+d1+'_en.bin', 'tm_'+ d1 +'_'+ d1 +'.p', 'tm_'+ d1 +'_'+ d1 +'.p'))
    #
    # for p in combinations(dimensions,2):
    #     de = str(p[0])
    #     en = str(p[1])
    #     tmFilenames.add(('mono_'+de+'_de.bin', 'mono_'+en+'_en.bin', 'tm_'+de+'_'+en+'.p', 'tm_'+en+'_'+de+'.p'))


    tmFilenames = [('mono_800_de.bin', 'mono_400_en.bin', 'tm_800_400.p', 'mono_400_de.bin', 'mono_800_en.bin', 'tm_800_400.p')]

    # TODO: this are debug param values. Comment.
    tmFilenames = [('mono_800_de.bin', 'mono_200_en.bin', 'tm_800_200.p', 'mono_200_de.bin', 'mono_800_en.bin', 'tm_800_200.p')]
    directPath = 'models/de-en/'
    inversePath = 'models/en-de/'
    modelsPath = 'models/'

    for t in tmFilenames:
        deModel = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[0],binary=True)
        enModel = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[1],binary=True)
        directMatrix = pickle.load(open(directPath + t[2]))
        deModelInverse = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[3],binary=True)
        enModelInverse = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[4],binary=True)
        inverseMatrix = pickle.load(open(inversePath + t[5]))

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

        corpus = PlaintextCorpusReader('./', f, encoding="utf-8")
        oovs = set(corpus.words())

        totalOOV = len(oovs)
        foutLog.write('Processing file ' + f + '\n')
        foutLog.write('Total number of unknown words: ' + str(totalOOV) + '\n')
        for oov in oovs:
            try:
                # get representations
                deRep = deModel[oov]
                # construct probabilities
            except KeyError:
                noTranslationWords.add(oov)
                continue

            calculatedRep = np.dot(directMatrix, deRep)
            candidates = enModel.most_similar(positive=[calculatedRep], topn=nDirectCandidates)

            # totalProb = 0
            normalizedCandidates = set()
            for wCand, _ in candidates:
                # totalProb += wCos
                # directProb = enModel.similarity(calculatedRep, wCand)
                directProb = np.dot(gensim.matutils.unitvec(calculatedRep), gensim.matutils.unitvec(enModel[wCand]))
                try:
                    inverseRep = enModelInverse[wCand]
                except KeyError:
                    inverseProb = 0
                    foutLog.write('Couldnt find inverse candidate representation for OOV: ' + oov + ' inv.Cand: ' + wCand + '\n')

                inverseCalculatedRep = np.dot(inverseMatrix, inverseRep)
                # inverseCandidate = deModelInverse.most_similar(positive=[inverseCalculatedRep], topn=nInverseCandidates)
                # inverseProb = deModelInverse.similarity(inverseCandidate[0][0], oov)
                # inverseProb = deModelInverse.similarity(inverseCalculatedRep, oov)
                inverseProb = np.dot(gensim.matutils.unitvec(inverseCalculatedRep), gensim.matutils.unitvec(deModelInverse[oov]))

                normalizedCandidates.add((wCand, directProb, inverseProb))
            for wCand, directProb, inverseProb in normalizedCandidates:
                convOOV = '##' + oov
                sep = '|||'
                alig = '0-0'
                # directProb = str(wCos / float(totalProb))
                directProb = str(directProb)
                inverseProb = str(inverseProb)
                line = ' '.join([convOOV,sep,wCand,sep,inverseProb, inverseProb,directProb,directProb,sep,alig,sep,sep,sep,'\n'])
                foutPhraseTable.write(line)

        foutLog.write('Total words translated: ' + str(totalOOV - len(noTranslationWords)) + '\n')
        for w in set(oovs) - set(noTranslationWords):
            foutLog.write('Translated word: ' + w + '\n')

        for w in noTranslationWords:
            foutLog.write('Untranslated word: ' + w + '\n')

        foutLog.close()
        # save second table
        foutPhraseTable.close()

