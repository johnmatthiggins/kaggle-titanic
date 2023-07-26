data:
	kaggle competitions download -c titanic
	mkdir data
	unzip titanic.zip -d data

train:
	./model.py

test:
	./model.py --test submission.csv


submit:
	kaggle competitions submit -c titanic -f submission.csv -m "$(date)"
