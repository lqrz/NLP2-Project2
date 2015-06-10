for p in $(ls | grep '^wordsToReplace2')
	do
		echo "making replacements in test corpus: $p"
		cp corpus/testing/test.true.de test.true."$p".de
		for oov in $(cat "$p")
			do
				sed -i "s/\<$oov\>/##$oov/g" test.true."$p".de
			done
		echo "generating second table"
		python generateSecondTable.py /home/wechsler/NLP2-Project2/models/truecase/ "$p" /home/wechsler/NLP2-Project2/models/truecase/de-en/yandex/ /home/wechsler/NLP2-Project2/models/truecase/en-de/yandex/

		echo "retuning Moses"
	done
