import gzip
from nltk.tokenize import RegexpTokenizer
import glob
import sys


class CorpusReader():
    """
    Reads corpus from gzip file.
    """

    def __init__(self, file_name, max_limit=None):
        self.file = gzip.open(file_name, "rb")
        self.max_limit = max_limit


    def __iter__(self):
        """
        Generator that returns a list of tokens for each sentence.
        :return: list of tokens
        """
        self.file.seek(0)
        c = 0
        for line in self.file:
            c += 1
            if self.max_limit and c > self.max_limit:
                break
            yield line.decode("utf-8").split()

    def get_raw_corpus(self):
        """ Generator that returns one sentence each time as a string (untokenized)
        :return:one sentence (string)
        """
        self.file.seek(0)
        c = 0
        for line in self.file:
            c += 1
            if self.max_limit and c > self.max_limit:
                break
            yield line





if __name__ == "__main__":
    # cr = CorpusReader("data/europarl-v7.en.gz", max_limit=100)
    # for sent in cr:
    #     print sent


    # pre-process corpora:
    # normalizes all corpus files in directory specified as argument
    # usage:
    # python CorpusReader.py data
    # - tokenize
    # - remove punctuation

    tokenizer = RegexpTokenizer(r'\w+')
    for f in glob.glob(sys.argv[1] + "/*"):
        print f
        outfile = open(f + ".norm", "w")
        cr = CorpusReader(f)
        for sent in cr.get_raw_corpus():
            outfile.write(" ".join(tokenizer.tokenize(sent.decode("utf-8"))).encode("utf-8") + "\n")

        outfile.close()
