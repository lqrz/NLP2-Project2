__author__ = 'lqrz'

from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
import sys
from dictionary import get_translations
from word2vec import loadModel
from sklearn.linear_model import Ridge
import numpy as np
import pickle

if __name__ == '__main__':

    # ------------- Begin params -------------#
    nWords = 5  # Number of words to train translation matrix
    nTranslations = 1

    alpha = 0.1  # linear regression solver regularization param

    # modelPathEn = sys.argv[2] # path to model file (word2vec)
    # modelPathDe = sys.argv[3] # path to model file (word2vec)
    modelPathEn = 'models/mono_800_en.bin'
    modelPathDe = 'models/mono_200_de.bin'
    enRepDimension = 800 # English word2vec representation dimension
    deRepDimension = 200 # Deutsch word2vec representation dimension

    # path = sys.argv[1] # path to .norm file (enlgish)
    path = 'data/en/norm/lqrz.en'  # path to .norm file (enlgish)

    # ------------- End params -------------#

    idx = path.rfind('/') + 1
    folder = path[0:idx]
    filename = path[idx:]

    wordlist = PlaintextCorpusReader(folder, filename)

    # Construct frequency dictionary
    fd = FreqDist(wordlist.words())

    translations = dict()

    # Get most frequent words to train projection matrix
    mostFrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)[:nWords]

    # Load word2vec trained models
    gensimModelEn = loadModel(modelPathEn)
    gensimModelDe = loadModel(modelPathDe)

    # Instantiate linear regression structures
    X = np.empty((0,enRepDimension))
    Y = np.empty((0,deRepDimension))

    for word, _ in mostFrequentWords:

        # Get translation from thesaurus api
        translations[word] = get_translations(word)[:nTranslations]

        # Get representations for words
        wordRepEn = gensimModelEn[word]

        for trans in translations[word]:
            wordRepDe = gensimModelDe[trans]

        # Add samples to linear regression structures
        X = np.r_[X,wordRepEn[np.newaxis,:]]
        Y = np.r_[Y,wordRepDe[np.newaxis,:]]

    # Instantiate linear regression solver
    solver = Ridge(alpha, solver='lsqr')

    # Fit data
    solver.fit(X, Y)

    # Store translation matrix
    pickle.dump(solver.coef_,open('lrweights.p','w'))
