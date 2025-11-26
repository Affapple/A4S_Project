from typing import Literal
import uuid
import numpy as np
import torch
from torchvision import datasets, transforms
import pandas as pd
import pytest

from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metric_registries.model_metric_registry import ModelMetric
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.service.model_factory import load_model

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
    Model,
    ModelConfig,
    ModelFramework,
    ModelTask,
)

from tests.save_measures_utils import save_measures
from tests.save_plot import draw_plot

MODEL_PATH = "mnist_cnn.pt"

def _get_mnist_dataset(train: bool = True) -> pd.DataFrame:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(root='./tests/data', train=train, download=True, transform=transform)
    pd_dataset = pd.DataFrame({
        'images': [dataset[i][0].numpy() for i in range(len(dataset))],
        'labels': [dataset[i][1] for i in range(len(dataset))]
    })
    return pd_dataset

@pytest.fixture
def train_dataset():
    return _get_mnist_dataset(train=True)

@pytest.fixture
def data_shape() -> DataShape:
    data_shape = {
        "features": [
            Feature(
                pid=uuid.uuid4(),
                name="images",
                feature_type=FeatureType.FLOAT,
                min_value=0,
                max_value=1,
            )
        ],
        "target": Feature(
            pid=uuid.uuid4(),
            name="labels",
            feature_type=FeatureType.INTEGER,
            min_value=0,
            max_value=9,
        ),
        "date": None
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def ref_dataset(train_dataset: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = train_dataset
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=ref_dataset,
    )


@pytest.fixture
def functional_model() -> TabularClassificationModel:
    model_config = ModelConfig(
        path=f"./tests/data/{MODEL_PATH}",
        framework=ModelFramework.TORCH,
        task=ModelTask.CLASSIFICATION,
    )

    model = load_model(model_config)
    if not isinstance(model, TabularClassificationModel):
        raise TypeError
    return model


def test_interpretability_metric_is_present():
    assert model_metric_registry._functions.get("local_lipschitz_estimate") is not None

@pytest.fixture
def testing_examples(data_shape: DataShape) -> dict[str, Dataset]:
    # load previously created datasets
    import pickle
    metric_testing_dataset: dict[str, dict[str, np.ndarray]] = pickle.load(open("./tests/data/metric_testing_dataset.pkl", "rb"))

    pd_adv_examples = {
        "images": metric_testing_dataset["adv_examples"]["x"],
        "labels": metric_testing_dataset["adv_examples"]["y"]
    }

    pd_well_classified = {
        "images": metric_testing_dataset["well_classified"]["x"],
        "labels": metric_testing_dataset["well_classified"]["y"]
    }
    pd_wrongly_classified = {
        "images": metric_testing_dataset["wrongly_classified"]["x"],
        "labels": metric_testing_dataset["wrongly_classified"]["y"]
    }

    return {
        "adv_examples": Dataset(
            pid=uuid.uuid4(),
            shape=data_shape,
            data=pd_adv_examples
        ),
        "well_classified": Dataset(
            pid=uuid.uuid4(),
            shape=data_shape,
            data=pd_well_classified
        ),
        "wrongly_classified": Dataset(
            pid=uuid.uuid4(),
            shape=data_shape,
            data=pd_wrongly_classified
        )
    }


@pytest.mark.parametrize("current_dataset", ["adv_examples", "well_classified", "wrongly_classified"])
def test_data_metric_registry_contains_evaluator(
    current_dataset: Literal["adv_examples", "well_classified", "wrongly_classified"],
    testing_examples: dict[str, Dataset],
    data_shape: DataShape,
    ref_model: Model,
    functional_model: TabularClassificationModel,
):
    dataset_df = testing_examples[current_dataset]
    metric_name = "local_lipschitz_estimate"
    metric_func = model_metric_registry.get_functions()[metric_name]


    print(current_dataset)
    measures = metric_func(
        data_shape, ref_model, dataset_df, functional_model
    )
    assert len(measures) > 0

    measure_name = f"{metric_name}-{current_dataset}"
    save_measures(measure_name, measures)