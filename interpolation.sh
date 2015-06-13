#!/bin/bash
for p in $(ls | grep '^all.*txt$')
	do
		if [ $1 -gt 0 ]
		then
			mkdir -p interpolation
			cp $p interpolation/
			cd interpolation/
			PHRASETABLE="phrase-table-interpolation"
		fi

		# STEP 1: GENERATE 2ND TABLE
		if [ $1 -le 1 ]
		then

			echo "generating second table"

			python ../generateSecondTable.py /home/wechsler/NLP2-Project2/models/truecase/ $p /home/wechsler/NLP2-Project2/models/truecase/de-en/yandex/ /home/wechsler/NLP2-Project2/models/truecase/en-de/yandex/

			mv "secondPhraseTable$p" $PHRASETABLE
			echo "compressing phrase table"
			gzip $PHRASETABLE
		fi

		# STEP 2: RETUNE MOSES
		if [ $1 -le 2 ]
		then
			echo "retuning Moses"
			MOSESFILE="mosesInterpolation.ini"
			cp /home/quiroz/NLP2-Project2/corpus/truecased/5k/train/model/$MOSESFILE $MOSESFILE
#			nohup nice /home/quiroz/NLP2-Project2/mosesdecoder/scripts/training/mert-moses.pl /home/quiroz/NLP2-Project2/corpus/tuning/true.de /home/quiroz/NLP2-Project2/corpus/tuning/true.en /home/quiroz/NLP2-Project2/mosesdecoder/bin/moses $MOSESFILE --decoder-flags="-threads 4" --mertdir /home/quiroz/NLP2-Project2/mosesdecoder/bin &> mert.out &
		fi
	done
