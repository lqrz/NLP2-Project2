import gensim





if __name__ == "__main__":
    from CorpusReader import CorpusReader
    cr = CorpusReader("data/europarl-v7.en.gz")


    model = gensim.models.Word2Vec(cr, min_count=1, size=400, workers=2)



