# Translation using baseline phrase table
echo "Translating test corpus"
/home/quiroz/NLP2-Project2/mosesdecoder/bin/moses -f /home/quiroz/NLP2-Project2/tuning/baseline/mert-work/moses.ini < /home/quiroz/NLP2-Project2/corpus/testing/test.true.de > /home/quiroz/NLP2-Project2/testing/natural/baseline.translated.en 2> /home/quiroz/NLP2-Project2/testing/natural/baseline.out

PID=$!

wait PID
# Bleu score using baseline phrase table
echo "Computing BLEU score"
/home/quiroz/NLP2-Project2/mosesdecoder/scripts/generic/multi-bleu.perl -lc /home/quiroz/NLP2-Project2/corpus/testing/test.true.en < /home/quiroz/NLP2-Project2/testing/natural/baseline.translated.en > naturalBaseline.score
