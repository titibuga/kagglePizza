Authors: Victor Sanches Portella
	 Allen Paramesh


## HOW TO USE


To use the code, first make sure to download the training and
test data from https://www.kaggle.com/c/random-acts-of-pizza with
the names "test.json" and "train.json", and make sure they are
in the same folder as the scripts.

To run the exploratory analysin, use the bash command:

$ python exploratory.py

To run the model, run the command:

$ python script.py

The last one will print the cross-validation AUC score and generate
a prediction of probabilities for the test file and will write these
prediction in the form "req_id prediction" in the file "testpredict.csv".

To change which model is being used go to the line 174 and choose which
model to use.

When using Random Forests to print the OOB score uncomment the
lines 264 and 265
