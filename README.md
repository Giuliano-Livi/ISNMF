In the folder "dataset" are present the dataset containing the real signals extracted.

In the folder "syntethic" are present the synergies taken from the Ninapro dataset.

In the file "ISNMF.py" there is the implementation of the ISNMF algorithm tested on simple signals.

In the file "ISNMF_synthetic_data.py" there is the implementation of the ISNMF algorithm tested on synthetic synergies.

In the file "ISNMF_real_dataset" there is the implementation of the ISNMF algorithm tested on the real datasets.

In the file "ISNMF_incremental_approach.py" there is the testing of the ISNMF algorithm training it with the real dataset and then simulating a shift of the brecelet and using the algorithm again.

In the file "ISNMF_final_test.py" there is the testing of the algorithm using some custom synthetic data that simulate real muvements, in particular there is the evaluation of the RMSE value given by the comparison between the original H matrix using which the input is created and the extracted H matrix.

In the file "representation_of_W.py" there are the graphical representation of the matrices W and H used for the testing of thee algorithm min the "ISNMF_final_test.py" file.
