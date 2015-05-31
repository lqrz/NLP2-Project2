'''
@author: lqrz
'''
import os
import sys
import re
from os import walk
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


def getPCASamples(fname):
    #240 each class
    m = np.zeros([480,4])
    
    features = dict()
    
    f = open(constants.PREPROCESSFEATURESFILENAME,'r')
    for l in f:
        l = l.strip()
        if l=='':
            continue
        q1 = l.split('\t')[0]
        q2 = l.split('\t')[1]
        f1 = l.split('\t')[2]
        f2 = l.split('\t')[3]
        f3 = l.split('\t')[4]
        f4 = l.split('\t')[5]
        features[(q1,q2)] = [f1, f2, f3, f4]
    
    
    f = open(constants.FOLDEDDATA_DIRECTORY+fname,'r')
    i=0
    for l in f:
        l = l.strip()
        if l==' ':
            continue
        q1 = l.split('\t')[1]
        q2 = l.split('\t')[2]
        m[i,:] = np.array(features[(q1,q2)])
        i += 1
    
    return m


if __name__ == '__main__':
    n_components = 3

    modelPathEn = 'models/mono_800_en.bin'
    modelPathDe = 'models/mono_200_de.bin'

    # Load word2vec trained models
    gensimModelEn = loadModel(modelPathEn)
    gensimModelDe = loadModel(modelPathDe)

    en_layer_size = gensimModelEn.layer1_size
    de_layer_size = gensimModelDe.layer1_size

    weights = pickle.load(open('lrweights.p','r'))

    path = 'data/en/norm/lqrz.en'

    idx = path.rfind('/') + 1
    folder = path[0:idx]
    filename = path[idx:]

    wordlist = PlaintextCorpusReader(folder, filename)
    # Construct frequency dictionary
    fd = FreqDist(wordlist.words())
    # Get most frequent words to train projection matrix
    mostFrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)[:20]
    mostInfrequentWords = sorted(fd.items(), key=lambda x: x[1], reverse=True)[-20:]
    frequentWords = [random.choice(mostFrequentWords) for _ in range(4)]
    infrequentWords = [random.choice(mostInfrequentWords) for _ in range(4)]

    samples = np.empty((8,de_layer_size))

    for i, word in enumerate(frequentWords):
        trans = get_translation_yandex(word[0])
        samples[i*2,:] = gensimModelDe[trans]
        samples[i*2+1,:] = np.dot(weights,gensimModelEn[word[0]])

    # for (_, _, filenames) in walk(constants.FOLDEDDATA_DIRECTORY):
    #     fnames = [re.match(r'fold[01234]_sorted',fn).group() for fn in filenames if re.match(r'fold[01234]_sorted',fn)]
    #
    # for i,fname in enumerate(fnames):
    #     samples = getPCASamples(fname)
    #     all_samples = # (3,40)
    #     assert samples.shape == (480,4), "The matrix has not the dimensions 3x40"
        
    if n_components==2:
        sklearn_pca = sklearnPCA(n_components=n_components)
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
        plt.title('Matrix translation approximation PCA')

        plt.show()
    elif n_components==3:
        sklearn_pca = sklearnPCA(n_components=n_components)
        sklearn_transf = sklearn_pca.fit_transform(samples)

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10
        ax.plot(sklearn_transf[0::2,0], sklearn_transf[0::2,1],\
            sklearn_transf[0::2,2], 'o', markersize=8, color='blue', alpha=0.5, label='Dictionary')
        ax.plot(sklearn_transf[1::2,0], sklearn_transf[1::2,1],\
            sklearn_transf[1::2,2], '^', markersize=8, alpha=0.5, color='red', label='Matrix')

        plt.title('Matrix translation approximation PCA')
        ax.legend(loc='upper right')
        # plt.legend()
        plt.show()