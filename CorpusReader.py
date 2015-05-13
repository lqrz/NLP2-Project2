import gzip


class CorpusReader():
    """
    Reads corpus from gzip file.
    """

    def __init__(self, file_name):
        self.file = gzip.open(file_name, "r")


    def __iter__(self):
        self.file.seek(0)
        for line in self.file:
            yield line.split()





if __name__ == "__main__":
    cr = CorpusReader("data/europarl-v7.en.gz")
    for sent in cr:
        print sent


