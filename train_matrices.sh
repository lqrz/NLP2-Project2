for endim in 200 400 800
do
  for dedim in 200 400 800
  do
    echo "Working on EN=${endim}, DE=${dedim}"
    echo "python translationMatrix2.py models/truecase/top10ken_trans_yandex.txt models/truecase/mono_${endim}_en.bin models/truecase/mono_${dedim}_de.bin models/truecase/tm_${endim}_${dedim}.p > results/log_${endim}_${dedim}.txt"
  done
done
echo "done"