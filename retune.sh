for p in $(ls | grep '^wordsToRep')
	do
		cp corpus/testing/test.true.de test.true."$p".de
		for oov in $(cat "$p")
			do
				sed -i "s/\<$oov\>/##$oov/g" test.true."$p".de
			done
	done
