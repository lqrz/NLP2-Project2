#!/bin/bash
for p in $(ls | grep '^wordsToReplace7')
	do
		NEWCORPUS="test.true.70.de"
		PHRASETABLE="phrase-table-70.gz"

		echo $NEWCORPUS
		echo "making replacements in test corpus: $p"

		cp corpus/testing/test.true.de $NEWCORPUS

		for oov in $(cat "$p")
			do
				sed -i "s/\<$oov\>/##$oov/g" $NEWCORPUS
			done
		echo "generating second table"

		python generateSecondTable.py /home/wechsler/NLP2-Project2/models/truecase/ "$p" /home/wechsler/NLP2-Project2/models/truecase/de-en/yandex/ /home/wechsler/NLP2-Project2/models/truecase/en-de/yandex/

		mv "secondPhraseTable$p.gz" $PHRASETABLE

		echo "retuning Moses"
	done
