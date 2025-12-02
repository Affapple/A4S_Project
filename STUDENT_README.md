# Student
***Name***: João Paulo Mendes Gaspra
***Student Number***: 025605833A

# Project

## Github Repository
[Project Implementation Github Repository](https://github.com/Affapple/A4S_Project)\
https://github.com/Affapple/A4S_Project

## 1st Milestone 
Refer to the file `Milestone1.txt`

## Metric Testing
Refer to the folder `A4S_metric_test` and the respective README

## Project Tree
The following files were edited during the project development
```
(N) -> New
(E) -> Edited
├── a4s_eval
│   ├── data_model
│   │   ├── evaluation.py (E)
│   ├── metrics
│   │   ├── model_metrics
│   │   │   └── interpretability_metric.py (N)
├── A4S_metric_tests (N)
│   ├── analysis.ipynb
│   ├── data
│   │   └── MNIST
│   ├── Lipschitz.py
│   ├── main.py
│   ├── metric_testing_dataset.pkl
│   ├── mnist_cnn.pt
│   ├── model.py
│   ├── README.md
│   ├── requirements.txt
│   ├── test_dataset.ipynb
│   └── train.ipynb
├── Milestone1.txt (N)
├── STUDENT_README.md (N)
├── tests
│   ├── data
│   │   ├── measures
│   │   │   └── (...)
│   │   ├── metric_testing_dataset.pkl
│   │   ├── mnist_cnn.pt
│   ├── metrics
│   │   ├── model_metrics
│   │   │   └── test_interpretability.py (N)
```
`a4s_eval/data_model_evaluation.py` file was edited when it shouldn't have been, however I felt some type of change to be of extreme importance. As it its, it only accepts tabular data, however this goes against my preposition, I couldn't find a way to load images into the project, and according to the model metric definition, tabular data is mandatory (Dataframe is strictly tabular), therefore I did the following change
```diff
class Dataset(BaseModel):
    pid: uuid.UUID
    shape: DataShape
-    data: pd.DataFrame | None = None
+    data: pd.DataFrame | dict[str, np.ndarray] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
```
With this change, I believe that nothing in the project changed, however it gave more freeedom to the data structure that can be passed to the metric.\
This should not be a permanent change, as differences will arise, but allowed for me to develop my metric given the time limits imposed.