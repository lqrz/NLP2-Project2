__author__ = 'rwechsler'
import codecs
import pickle



def collect_translations(list_infile_name, lex_infile_name, pickle_outfile_name=None):

    # read top 10k words

    list_infile = codecs.open(list_infile_name, "r", "utf-8")

    transl_dict = dict()


    for line in list_infile:
        transl_dict[line.strip()] = []


    infile = codecs.open(lex_infile_name, "r", "utf-8")

    for line in infile:
        els = line.strip().split()
        try:
            transl_dict[els[1]].append((els[0], float(els[2])))
        except:
            pass

    infile.close()

    if pickle_outfile_name:

        pickle.dump(transl_dict, open(pickle_outfile_name, "wb"))

    return transl_dict



def produce_translation_file(outfile_name, pickle_file_name=None, translation_dict=None, list_infile=None, list_infile_name=None):
    # produce translation file

    if pickle_file_name:
        transl_dict = pickle.load(open(pickle_file_name, "rb"))

    else:
        transl_dict = translation_dict

    outfile = codecs.open(outfile_name, "w", "utf-8")

    if list_infile_name:
        list_infile = codecs.open(list_infile_name, "r", "utf-8")
    else:
        list_infile.seek(0)

    for line in list_infile:
        word = line.strip()
        try:
            tmp = sorted(transl_dict[word], key=lambda tup:tup[1], reverse=True)[:5]
            if tmp[0][0] == "NULL":
                outfile.write(word + "\t" + tmp[1][0] + "\n")
            else:
                outfile.write(word + "\t" + tmp[0][0] + "\n")

        except:
            pass

    outfile.close()

    list_infile.close()
