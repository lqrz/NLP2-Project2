#!/bin/bash
for p in $(ls | grep '^wordsToReplace2')
	do
		if [ $1 -gt 0 ]
		then
			mkdir -p 20
			cd 20/
		fi

		# STEP 0: MAKE REPLACEMENTS IN CORPUS
		if [ $1 -eq  0 ]
		then
			mkdir -p 20
			cp $p 20/
			cd 20/
			NEWCORPUS="test.true.20.de"
			PHRASETABLE="phrase-table-20.gz"

			echo "making replacements in test corpus: $p"

			cp ../corpus/testing/test.true.de $NEWCORPUS

			for oov in $(cat $p)
				do
					sed -i "s/\<$oov\>/##$oov/g" $NEWCORPUS
				done
		fi

		# STEP 1: GENERATE 2ND TABLE
		if [ $1 -le 1 ]
		then

			echo "generating second table"

			python ../generateSecondTable.py /home/wechsler/NLP2-Project2/models/truecase/ $p /home/wechsler/NLP2-Project2/models/truecase/de-en/yandex/ /home/wechsler/NLP2-Project2/models/truecase/en-de/yandex/

			mv "secondPhraseTable$p.gz" $PHRASETABLE
		fi

		# STEP 2: RETUNE MOSES
		if [ $1 -le 2 ]
		then
			echo "retuning Moses"
			MOSESFILE="moses20.ini"
			cp /home/quiroz/NLP2-Project2/corpus/truecased/5k/train/model/$MOSESFILE $MOSESFILE
			nohup nice /home/quiroz/NLP2-Project2/mosesdecoder/scripts/training/mert-moses.pl /home/quiroz/NLP2-Project2/corpus/tuning/true.de /home/quiroz/NLP2-Project2/corpus/tuning/true.en /home/quiroz/NLP2-Project2/mosesdecoder/bin/moses $MOSESFILE --decoder-flags="-threads 4" --mertdir /home/quiroz/NLP2-Project2/mosesdecoder/bin &> mert.out &
		fi
	done
