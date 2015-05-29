__author__ = 'rwechsler'
import codecs
import gensim
import numpy as np
import pickle
from sklearn.linear_model import Ridge


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

            if found_transl:
                found_similarities.append(float(similarity_sum) / found_transl)
            else:
                found_similarities.append(0)

    return found_similarities, candidate_translations

def calculate_accuracy(similarities):
    total = 0
    correct = 0
    sum = 0
    for sim in similarities:
        total += 1
        if sim > 0:
            correct += 1
            sum += sim

    accuracy = float(correct) / total if total > 0 else 0
    average_sim = float(sum) / correct if correct > 0 else 0

    return accuracy, average_sim


if __name__ == '__main__':

    # ------------- Begin params -------------#
    matrix_file = "models/800_200_tm.p"
    nWords = 5000  # Number of words to train translation matrix. Set to None if fixed sample size
    # nSamples = None # Fixed sample size, set to None if it doesn't matter.
    nTranslations = 1 # (max) number of translation variants per word (default=1)

    dict_file = "models/truecase/top10ken_trans_yandex.txt"

    alpha = 0.1  # linear regression solver regularization param

    modelPathEn = 'models/truecase/mono_800_en.bin'
    modelPathDe = 'models/truecase/mono_200_de.bin'

    # Evaluation
    n_test_samples = 100 # number of test samples (subsequent samples not used in training)
    n_test_translations = 1 # number of translation variants to be considered as gold standard
    k_nearest = 5 # number of k nearest words of translated vector to be considered.


    # ------------- End params -------------#



    samples = get_dict_samples(dict_file)

    train_samples = samples[:nWords]
    test_samples = samples[nWords:nWords+n_test_samples]


    # Load word2vec trained models
    print "Loading models."
    gensimModelEn = gensim.models.Word2Vec.load_word2vec_format(modelPathEn, binary=True)
    gensimModelDe = gensim.models.Word2Vec.load_word2vec_format(modelPathDe, binary=True)

    en_layer_size = gensimModelEn.layer1_size
    de_layer_size = gensimModelDe.layer1_size

    # Instantiate linear regression structures
    # Create max. size matrices and fill them up
    # X = samples x layer_size_en
    # Y = samples x layer_size_de
    X = np.zeros((nWords * nTranslations, en_layer_size))
    Y = np.empty((nWords * nTranslations, de_layer_size))


    sample_count = 0
    word_count = 0
    for word, translations in train_samples:
        # Get representations for words
        wordRepEn = gensimModelEn[word]
        transl_accepted = False
        for trans in translations[:nTranslations]:
            try:
                wordRepDe = gensimModelDe[trans]

                X[sample_count] = wordRepEn
                Y[sample_count] = wordRepDe
                sample_count += 1
                transl_accepted = True
            except KeyError:
                print "Translation not found: ", trans
        word_count += int(transl_accepted)


    # Reduce matrices to actual sample size
    X = X[:sample_count, :]
    Y = Y[:sample_count, :]

    print str(sample_count) + " samples from " + str(word_count) + " words."


    print "Learning translation matrix"
    # Instantiate linear regression solver
    solver = Ridge(alpha, solver='lsqr')

    # Fit data
    solver.fit(X, Y)

    # translation_matrix
    translation_matrix = solver.coef_

    print "Saving matrix"
    # Store translation matrix
    pickle.dump(translation_matrix,open(matrix_file,'w'))

    print "Evaluating"
    # evaluation
    similarities, predicted_translations = evaluate(test_samples, gensimModelEn, gensimModelDe, translation_matrix, k_nearest=k_nearest, n_translations=n_test_translations)
    accuracy, average_sim = calculate_accuracy(similarities)

    for i, (s, t) in enumerate(test_samples):
        print "Test word: ", s
        print "Gold: ", t
        print "Cand: ", predicted_translations[i]
        print "Sim: ", similarities[i]

    print "Accuracy: ", accuracy
    print "Average similarity: ", average_sim









