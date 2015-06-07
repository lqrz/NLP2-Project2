'''
@author: lqrz
'''
import os
import sys
from sklearn.decomposition import PCA as sklearnPCA
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import pickle
from word2vec import loadModel
import random
from nltk import FreqDist
from nltk.corpus import PlaintextCorpusReader
from dictionary import get_translation_yandex

def plotGraph(samples, dimensions, typ='Frequent words'):
    if dimensions == '2D':
        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        plt.plot(sklearn_transf[0::2,0],sklearn_transf[0::2,1],\
             'o', markersize=7, color='blue', alpha=0.5, label='Dictionary')
        plt.plot(sklearn_transf[1::2,0], sklearn_transf[1::2,1],\
             '^', markersize=7, color='red', alpha=0.5, label='Matrix')

        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
    #     plt.xlim([-4,4])
    #     plt.ylim([-4,4])
        plt.legend()
        plt.title(typ+' matrix translation approximation PCA')

    elif dimensions == '3D':
        sklearn_pca = sklearnPCA(n_components=3)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10
        ax.plot(sklearn_transf[0::2,0], sklearn_transf[0::2,1],\
            sklearn_transf[0::2,2], 'o', markersize=8, color='blue', alpha=0.5, label='Dictionary')
        ax.plot(sklearn_transf[1::2,0], sklearn_transf[1::2,1],\
            sklearn_transf[1::2,2], '^', markersize=8, alpha=0.5, color='red', label='Matrix')

        plt.title(typ+' matrix translation approximation PCA')
        ax.legend(loc='upper right')

    
    plt.savefig("%s-%s.png" % (typ,dimensions), bbox_inches='tight',dpi=200)
    plt.close()

    return True

def constructSamplesAndPlot(words, de_layer_size, gensimModelDe, gensimModelEn, weights, typ='Frequent words'):
    global lowercased

    samples = np.empty((8,de_layer_size))

    print('Getting samples...')

    # for i, word in enumerate(words):
    c = 0
    while c < 4:
        word = random.choice(words)
        try:
            trans = get_translation_yandex(word[0])
            if lowercased:
                trans = trans.lower()
            samples[c*2,:] = gensimModelDe[trans]
            samples[c*2+1,:] = np.dot(weights, gensimModelEn[word[0]])
            c += 1
        except:
            continue
        #
        # # if word[0] in gensimModelEn.keys():
        #     trans = get_translation_yandex(word[0])
        #     if trans in gensimModelDe.keys():
        #         samples[i*2,:] = gensimModelDe[trans]
        #         samples[i*2+1,:] = np.dot(weights,gensimModelEn[word[0]])
        #         c += 1

    plotGraph(samples,dimensions='2D',typ=typ)
    plotGraph(samples,dimensions='3D',typ=typ)

    return True


if __name__ == '__main__':
    # n_components = 3

    lowercased = True

    modelPathEn = 'models/lowercase/mono_800_en.bin'
    modelPathDe = 'models/lowercase/mono_200_de.bin'

    path = 'data/en/norm/lqrz.en'

    weightsPickle = 'lrweights.p'



    if len(sys.argv) == 6:

        if sys.argv[1]:
            # word2vec En model
            modelPathEn = sys.argv[1]
            if not os.path.exists(modelPathEn):
                print 'Path does not exist ', modelPathEn

        if sys.argv[2]:
            # word2vec De model
            modelPathDe = sys.argv[2]
            if not os.path.exists(modelPathDe):
                print 'Path does not exist ', modelPathDe

        if sys.argv[3]:
            # training corpus (source corpus)
            path = sys.argv[3]
            if not os.path.exists(path):
                print 'Path does not exist ', path

        if sys.argv[4]:
            # translation matrix pickle
            weightsPickle = sys.argv[4]
            if not os.path.exists(weightsPickle):
                print 'Path does not exist ', weightsPickle

        if sys.argv[5] == 'lowercased':
            # translation matrix pickle
            lowercased = True
        elif sys.argv[5] == 'truecased':
            lowercased = False
        else:
            print 'Enter -option lowercased or truecased'
            exit()

    elif len(sys.argv) > 1 and len(sys.argv) < 6:
        print '-En word2vec model -De word2vec model -source training corpus -translation matrix pickle -lowercased\\truecased'
        exit()


    # Load word2vec trained models
    print('Loading word2vec models...')
    gensimModelEn = loadModel(modelPathEn)
    gensimModelDe = loadModel(modelPathDe)

    en_layer_size = gensimModelEn.layer1_size
    de_layer_size = gensimModelDe.layer1_size

    print('Loading translation matrix pickle...')
    weights = pickle.load(open(weightsPickle, 'r'))


    idx = path.rfind('/') + 1
    folder = path[0:idx]
    filename = path[idx:]

    print('Getting corpus...')
    wordlist = PlaintextCorpusReader(folder, filename)
    # Construct frequency dictionary
    fd = FreqDist(wordlist.words())
    # Get most frequent words to train projection matrix
    mostFrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)[:20]
    mostInfrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)[-20:]

    # frequentWords = [random.choice(mostFrequentWords) for _ in range(4)]
    # infrequentWords = [random.choice(mostInfrequentWords) for _ in range(4)]

    constructSamplesAndPlot(mostFrequentWords, de_layer_size, gensimModelDe, gensimModelEn, weights, typ='Frequent words')
    constructSamplesAndPlot(mostInfrequentWords, de_layer_size, gensimModelDe, gensimModelEn, weights, typ='Infrequent words')

