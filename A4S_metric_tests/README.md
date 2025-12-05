# How to use this folder
To use this folder, you should not use the virtual environment from a4s! \
To use this folder you should:
- Create a virual environment by running `python3 -m venv .venv`
- Activate the virtual environment by running `source .venv/bin/activate`
- Install the required dependecies by running `pip install -r requirements.txt`
- Activate the environment by running `source .venv/bin/activate`, it can be deactivated at any time by running `deactivate`

Now you should be able to run the notebooks.

# Testing forlder organization
This testing folder is divided into tree important files:
- `train.ipynb`: Meant to train a CNN model to predict the written number using MNIST handwritten images as a dataset
- `test_dataset.ipynb`: Meant to get examples to test the metric I implemented in A4S for both MNIST and Income models:
  - ***Original (Adv)***: Getting 10 well classified images and saving them
  - ***Adversarial***: Attacking the model to generate 10 adversarial examples beginning with the previously mentioned examples
  - ***Well Classified***: Getting other 10 well classified images
  - ***Wrongly Classified***: Getting 10 wrongly classified images
- `analysis.ipynb`: Notebooke where we analyse the obtained results

Both `Lipschitz.py` and `model.py` are auxiliary files where I defined the metric, for testing purposes and not for grading purposes, and the model, respectively.

# How to run

Firstly you have to get the trained model and its datasets, refer to `train.ipynb` and `test_dataset.ipynb` and copy the generated files to the mentioned directory.\
Then you have to run the script to run the tests of the project, run `./tests/run_metrics_test.sh`.\
If it fails for lack of permissions, add the permission to execute to the file: `chmod +x ./tests/run_metrics_test.sh`.
  - **Note**: the tests will return warnings when running them, I believe they can be safely ignored, however I chose to not supress them in case they aren't.

Now you should have the files necessary to run `analysis.ipynb` to analyse the results, refer to that file.