# How to use this folder
To use this folder, you should not use the virtual environment from a4s! \
To use this folder you should:
- Create a virual environment by running `python3 -m venv .venv`
- Activate the virtual environment by running `source .venv/bin/activate`
- Install the required dependecies by running `pip install -r requirements.txt`

Now you should be able to run the notebooks.

# Testing forlder organization
This testing folder is divided into tree important files:
- `train.ipynb`: Meant to train a CNN model to predict the written number using MNIST handwritten images as a dataset
- `test_dataset.ipynb`: Meant to get examples to test the metric I implemented in A4S by:
  - ***Original (Adv)***: Getting 10 well classified images and saving them
  - ***Adversarial***: Attacking the model to generate 10 adversarial examples beginning with the previously mentioned examples
  - ***Well Classified***: Getting other 10 well classified images
  - ***Wrongly Classified***: Getting 10 wrongly classified images
- `analysis.ipynb`: Notebooke where we analyse the obtained results

Both `Lipschitz.py` and `model.py` are auxiliary files where I defined the metric, for testing purposes and not for grading purposes, and the model, respectively.