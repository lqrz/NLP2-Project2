import gzip
import nltk


class CorpusReader():
    """
    Reads corpus from gzip file.
    """

    def __init__(self, file_name, max_limit=None):
        self.file = gzip.open(file_name, "r")
        self.max_limit = max_limit


    def __iter__(self):
        self.file.seek(0)
        c = 0
        for line in self.file:
            c += 1
            if self.max_limit and c > self.max_limit:
                break
            yield nltk.word_tokenize(line.decode("utf-8"))





if __name__ == "__main__":
    cr = CorpusReader("data/europarl-v7.en.gz", max_limit=100)
    for sent in cr:
        print sent


