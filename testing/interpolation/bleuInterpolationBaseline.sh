# Translation using baseline phrase table
/home/quiroz/NLP2-Project2/mosesdecoder/bin/moses -f /NLP2-Project2/tuning/baseline/mert-work/moses.ini < /home/quiroz/NLP2-Project2/corpus/testing/test.true.de > /home/quiroz/NLP2-Project2/testing/interpolation/baseline.translated.en 2> /home/quiroz/NLP2-Project2/testing/interpolation/baseline.out
# Bleu score using baseline phrase table
/home/quiroz/NLP2-Project2/mosesdecoder/scripts/generic/multi-bleu.perl -lc /home/quiroz/NLP2-Project2/corpus/testing/test.true.en < /home/quiroz/NLP2-Project2/testing/interpolation/baseline.translated.en > interpolationBaseline.score
