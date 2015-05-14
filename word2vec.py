import gensim
from CorpusReader import CorpusReader

def train_vector_representations(corpus_file, output_file, min_count=5, vector_dim=400, workers=1):
    cr = CorpusReader(corpus_file)

    model = gensim.models.Word2Vec(cr, min_count=min_count, size=vector_dim, workers=workers)
    model.save(output_file)







if __name__ == "__main__":

    # train and save model
    train_vector_representations("data/europarl-v7.en.gz", "models/en.w2v", min_count=5, vector_dim=400, workers=3)

    # load model
    model = gensim.models.Word2Vec.load('models/en.w2v')






