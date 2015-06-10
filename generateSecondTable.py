__author__ = 'lqrz'
import sys
from itertools import combinations
import pickle
import gensim
import numpy as np
import codecs

if __name__ == '__main__':

    nCandidates = 10 # number of possible translations to retrieve from embedding space

    outputPhraseTablePath = 'secondPhraseTable'
    outputLogFilePath = 'secondTable.log'

    foutPhraseTable = codecs.open(outputPhraseTablePath, "w", "utf-8")
    foutLog = codecs.open(outputLogFilePath, "w", "utf-8")

    modelsPath = '/home/wechsler/NLP2-Project2/models/truecase/'
    if len(sys.argv) > 1:
        modelsPath = sys.argv[1]

    dimensions = [200,400,800]

    oovFilenames = ['oovs24.5998608212.p'] # TODO: Add pickle filenames

    directPath = modelsPath + 'de-en/yandex/'
    inversePath = modelsPath + 'en-de/'

    tmFilenames = set()
    modelFilenames = set()

    for d in dimensions:
        d1 = str(d)
        tmFilenames.add(('mono_'+d1+'_de.bin', 'mono_'+d1+'_en.bin', 'tm_'+ d1 +'_'+ d1 +'.p', 'tm_'+ d1 +'_'+ d1 +'.p'))

    for p in combinations(dimensions,2):
        de = str(p[0])
        en = str(p[1])
        tmFilenames.add(('mono_'+de+'_de.bin', 'mono_'+en+'_en.bin', 'tm_'+de+'_'+en+'.p', 'tm_'+en+'_'+de+'.p'))


    tmFilenames = [('mono_800_de.bin', 'mono_200_en.bin', 'tm_800_200.p', 'tm_200_800.p')]
    directPath = 'models/'
    inversePath = 'models/'
    modelsPath = 'models/'

    for i,t in enumerate(tmFilenames):
        deModel = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[0],binary=True)
        enModel = gensim.models.Word2Vec.load_word2vec_format(modelsPath+t[1],binary=True)
        directMatrix = pickle.load(open(directPath + t[2]))
        inverseMatrix = pickle.load(open(inversePath + t[3]))

    noTranslationWords = set()

    # load OOVs
    for f in oovFilenames:
        oovs = pickle.load(open(f,'rb'))
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
                    sep = '|||'
                    directProb = str(wCos / float(totalProb))
                    inverseProb = str(inverseProb)
                    line = ' '.join([oov,sep,wCand,sep,inverseProb, inverseProb,directProb,directProb,sep,sep,sep,sep,sep])
                    foutPhraseTable.write(line)


                # construct probabilities
            except:
                noTranslationWords.add(oov)
                continue

        foutLog('Total words translated: ' + str(totalOOV - len(noTranslationWords)) + '\n')
        for w in set(oovs) - set(noTranslationWords):
            foutLog('Translated word: ' + w + '\n')

        for w in noTranslationWords:
            foutLog('Untranslated word: ' + w + '\n')
        # save second table
        foutPhraseTable.close()