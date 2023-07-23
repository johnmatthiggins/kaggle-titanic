data:
	kaggle competitions download -c titanic
	mkdir data
	unzip titanic.zip -d data

test:
	./model.py --test results.csv

