# Translation using enhanced phrase table
/home/quiroz/NLP2-Project2/mosesdecoder/bin/moses -f /home/quiroz/NLP2-Project2/tuning/interpolation/mert-work/moses.ini < /home/quiroz/NLP2-Project2/corpus/testing/test.true.de > /home/quiroz/NLP2-Project2/testing/interpolation/interpolation.translated.en 2> /home/quiroz/NLP2-Project2/testing/interpolation/interpolation.out
# Bleu score using enhanced phrase table
/home/quiroz/NLP2-Project2/mosesdecoder/scripts/generic/multi-bleu.perl -lc /home/quiroz/NLP2-Project2/corpus/testing/test.true.en < /home/quiroz/NLP2-Project2/testing/interpolation/interpolation.translated.en > interpolationEnhanced.score
