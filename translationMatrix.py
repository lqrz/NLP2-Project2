__author__ = 'lqrz'

from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
import sys
from dictionary import get_translations
from word2vec import loadModel
from sklearn.linear_model import Ridge
import numpy as np

if __name__ == '__main__':

    # ------------- Begin params -------------#
    nWords = 5  # Number of words to train translation matrix
    nTranslations = 1
    alpha = 0.1  # linear regression solver regularization param
    # path = sys.argv[1] # path to .norm file (enlgish)
    # modelPath = sys.argv[2] # path to model file (word2vec)
    modelPath = '/'
    # path = '/home/lqrz/Desktop/fold/europarl.en.norm'  # path to .norm file (enlgish)
    path = 'data/en/norm/lqrz.en'  # path to .norm file (enlgish)
    enRepDimension = 800 # English word2vec representation dimension
    deRepDimension = 200 # Deutsch word2vec representation dimension
    # ------------- End params -------------#

    idx = path.rfind('/') + 1
    folder = path[0:idx]
    filename = path[idx:]

    wordlist = PlaintextCorpusReader(folder, filename)

    # Construct frequency dictionary
    fd = FreqDist(wordlist.words())

    # Get translations from api En -> De
    translations = dict()

    mostFrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)[:nWords]

    # Load word2vec trained models
    gensimModelEn = loadModel('models/mono_800_en.bin')
    gensimModelDe = loadModel('models/mono_200_de.bin')

    X = X = np.empty((0,enRepDimension))
    Y = X = np.empty((0,deRepDimension))

    for word, _ in mostFrequentWords:

        # Get translation from thesaurus api
        translations[word] = get_translations(word)[:nTranslations]

        # Get representations for words
        wordRepEn = gensimModelEn[word]

        for trans in translations[word]:
            wordRepDe = gensimModelDe[trans]

        X = np.r_[X,wordRepEn[np.newaxis,:]]
        Y = np.r_[Y,wordRepEn[np.newaxis,:]]

    # Instantiate linear regression solver
    solver = Ridge(alpha, solver='lsqr')

    # Fit data
    solver.fit(X, Y)

    # Store translation matrix
    solver.coef_
