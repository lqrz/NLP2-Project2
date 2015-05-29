import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/gensim')
from gensim import gensim
from CorpusReader import CorpusReader

def train_vector_representations(corpus_file, output_file, min_count=5, vector_dim=400, workers=1):
    cr = CorpusReader(corpus_file)

    model = gensim.models.Word2Vec(cr, min_count=min_count, size=vector_dim, workers=workers)
    model.save(output_file)

def loadModel(filename):
    return gensim.models.Word2Vec.load_word2vec_format(filename,binary=True)



if __name__ == "__main__":

    # train and save model
    dataFilename = "/home/wechsler/NLP-Project2/data/europarl-v7.en.gz"
    train_vector_representations(dataFilename, "models/en.w2v", min_count=5, vector_dim=400, workers=8)

    # load model
    model = gensim.models.Word2Vec.load('models/en.w2v')






