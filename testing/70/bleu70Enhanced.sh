# Translation using enhanced phrase table
echo "Translating test corpus"
/home/quiroz/NLP2-Project2/mosesdecoder/bin/moses -f /home/quiroz/NLP2-Project2/tuning/70/mert-work/moses.ini < /home/quiroz/NLP2-Project2/70/test.true.70.de > /home/quiroz/NLP2-Project2/testing/70/70.translated.en 2> /home/quiroz/NLP2-Project2/testing/70/70.out

PID=$!
wait PID

# Bleu score using enhanced phrase table
echo "Computing BLEU score"
/home/quiroz/NLP2-Project2/mosesdecoder/scripts/generic/multi-bleu.perl -lc /home/quiroz/NLP2-Project2/corpus/testing/test.true.en < /home/quiroz/NLP2-Project2/testing/70/70.translated.en > 70Enhanced.score
