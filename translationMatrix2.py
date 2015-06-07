__author__ = 'rwechsler'
import codecs
import gensim
import numpy as np
import pickle
from sklearn.linear_model import Ridge, RidgeCV
import sys
from collections import Counter


def get_dict_samples(dict_file):
    samples = []
    infile = codecs.open(dict_file, "r", "utf-8")
    for line in infile:
        els = line.strip().split("\t")
        samples.append((els[0], tuple(els[1:])))

    return samples


def translate_with_matrix(translation_matrix, vector):
    return np.dot(translation_matrix, vector)

def get_k_nearest_translations(vector, model, k=1):
    return model.most_similar(positive=[vector], topn=k)

def evaluate(test_samples, en_model, de_model, translation_matrix, k_nearest=1, n_translations=None):
    """
    Evaluates a translation matrix based on test samples.
    :param test_samples: List of test samples [(word1, (trans1, trans2, ...), ...]
    :param en_model: en gensim model (source)
    :param de_model: de gensim model (target)
    :param translation_matrix: translation matrix
    :param k_nearest: number of translations (k-nearest to approximated target vector) to be considered
    :param n_translations: max. number of gold translations to be considered
    :return: list of averaged similarities and list of candidate translations for each test sample
    """
    found_similarities = []
    candidate_translations = []
    ranks = []
    for word, translations in test_samples:
            # Get representations for words
            test_rep = en_model[word]
            translated_rep =  translate_with_matrix(translation_matrix, test_rep)
            predicted_translations = get_k_nearest_translations(translated_rep, de_model, k=k_nearest)
            candidate_translations.append(predicted_translations)
            predicted_translations, similarities = zip(*predicted_translations)
            found_transl = 0
            similarity_sum = 0
            for gold_trans in translations[:n_translations]:
                for i, pred_trans in enumerate(predicted_translations):
                    if gold_trans == pred_trans:
                        found_transl += 1
                        similarity_sum += similarities[i]
                        ranks.append(i+1)

            if found_transl:
                found_similarities.append(float(similarity_sum) / found_transl)
            else:
                ranks.append(0)
                found_similarities.append(0)

    return found_similarities, candidate_translations, ranks

def calculate_acc_avgsim(similarities, ranks):

    total = len(ranks)
    c = Counter(ranks)

    counts = [float(c.get(r, 0)) for r in range(1, 1+np.max(c.keys()))]
    accuracies = np.cumsum(counts) / float(total)

    average_sim = np.mean([s for s in similarities if s > 0])
    average_rank = np.mean([r for r in ranks if r > 0])

    return accuracies, average_sim, average_rank

def learn_translation_matrix(X, Y, alphas=[0.1, 1.0, 10.0]):
    # Instantiate linear regression solver
    solver = RidgeCV(alphas=alphas, cv=None, fit_intercept=True, scoring=None,
    normalize=False)

    # Fit data
    solver.fit(X, Y)

    print "Best alpha is ", solver.alpha_

    # translation_matrix
    return solver.coef_, solver.alpha_


def get_valid_samples(samples, n, model, n_translations=None):
    valid = []
    c = 0
    for i, sample in enumerate(samples):
        valid_trans = []
        for trans in sample[1][:n_translations]:
            try:
                model[trans]
                valid_trans.append(trans)
            except KeyError:
                pass

        if valid_trans:
            valid.append((sample[0], tuple(valid_trans)))
            c += 1

        if c == n:
            break

    return valid, samples[i+1:]

def get_train_test_sets(dict_file, n_translations, n_train_samples, n_test_samples, de_model):
    """
    Splits data into training and test samples.
    :param dict_file: File that contains translations
    :param n_translations: Top n translations to be considered.
    :param n_train_samples: Number of (valid) samples for training.
    :param n_test_samples: Number of (valid) samples for testing.
    :param de_model: de_model, required to see if translation is available as in vector space.
    :return: training and test samples
    """
    samples = get_dict_samples(dict_file)
    train_samples, rest = get_valid_samples(samples, n_train_samples, de_model, n_translations)
    train_samples = [(k, t) for (k,v) in train_samples for t in v]
    test_samples, _ = get_valid_samples(rest, n_test_samples, de_model)
    return train_samples, test_samples


if __name__ == '__main__':
    # Either runs parameters specified here or the ones in the command-line.

    # ------------- Begin params -------------#
    matrix_file = "models/tm_test.p"
    n_train_samples = 5000  # Number of words to train translation matrix.
    n_translations = 1 # (max) number of translation variants per word (default=1)

    dict_file = "models/truecase/top10ken_trans_yandex.txt" # file that contains en word and de translation(s)

    modelPathEn = 'models/truecase/mono_800_en.bin' # en word2vec model
    modelPathDe = 'models/truecase/mono_200_de.bin' # de word2vec model

    # Evaluation
    n_test_samples = 1000 # number of test samples (subsequent words not used in training)
    n_test_translations = 1 # number of translation variants to be considered as gold standard
    k_nearest = 5 # number of k nearest words of translated vector to be considered.
    # ------------- End params -------------#

    if len(sys.argv) > 1:
        try:
            dict_file, modelPathEn, modelPathDe, matrix_file = sys.argv[1:]
            print "Running with command-line arguments."
        except:
            print "Error arguments:"
            print "<dict_file> <model_en> <model_de> <matrix_output_file>"
    else:
        print "Running with parameters specified in file."


    # Load word2vec trained models
    print "Loading models."
    gensimModelEn = gensim.models.Word2Vec.load_word2vec_format(modelPathEn, binary=True)
    gensimModelDe = gensim.models.Word2Vec.load_word2vec_format(modelPathDe, binary=True)

    en_layer_size = gensimModelEn.layer1_size
    de_layer_size = gensimModelDe.layer1_size


    # Split training and test set
    # Makes sure that only valid samples (with vectors) are selected.
    train_samples, test_samples = get_train_test_sets(dict_file, n_translations, n_train_samples, n_test_samples, gensimModelDe)



    # Instantiate linear regression structures
    # Create max. size matrices and fill them up
    # X = samples x layer_size_en
    # Y = samples x layer_size_de
    X = np.zeros((len(train_samples), en_layer_size))
    Y = np.empty((len(train_samples), de_layer_size))


    for i, (word, translation) in enumerate(train_samples):
        # Get representations for words
        wordRepEn = gensimModelEn[word]
        wordRepDe = gensimModelDe[translation]
        X[i] = wordRepEn
        Y[i] = wordRepDe


    print "Training matrix with alpha cross-validation"
    translation_matrix, alpha = learn_translation_matrix(X, Y)

    print "Saving matrix"
    # Store translation matrix
    pickle.dump(translation_matrix,open(matrix_file,'w'))

    print "Evaluating"
    # evaluation
    similarities, predicted_translations, ranks = evaluate(test_samples, gensimModelEn, gensimModelDe,
                                                           translation_matrix, k_nearest=k_nearest,
                                                           n_translations=n_test_translations)
    r_acc, average_sim, average_rank = calculate_acc_avgsim(similarities, ranks)

    # print test results
    # comment out if not desired
    for i, (s, t) in enumerate(test_samples):
        print "Test word: ", s.encode("utf-8")
        print "Gold: ", t
        print "Cand: ", predicted_translations[i]
        print "Sim: ", similarities[i]

    # print final results
    print "----------"
    print "Samples train/test: ", n_train_samples, n_test_samples
    print "Alpha: ", alpha
    print "Ranked accuracies: ", r_acc
    print "Average similarity: ", average_sim
    print "Average rank: ", average_rank

