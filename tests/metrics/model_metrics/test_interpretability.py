from typing import Literal
import uuid
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

def dataset(train: bool = False):
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
def dataset_train():
    return dataset(train=True)

@pytest.fixture
def dataset_test():
    return dataset(train=False)

@pytest.fixture
def data_shape() -> DataShape:
    data_shape = {
        "features": [
            Feature(
                pid=uuid.uuid4(),
                name="images",
                feature_type=FeatureType.FLOAT,
                min_value=0,
                max_value=255,
            )
        ],
        "target": Feature(
            pid=uuid.uuid4(),
            name="label",
            feature_type=FeatureType.INTEGER,
            min_value=0,
            max_value=255,
        ),
        "date": None
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(dataset_test: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = dataset_test
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(dataset_train: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = dataset_train
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture
def ref_model(ref_dataset: Dataset) -> Model:
    """I needed the reference model, since I used pytorch model, I had to pass it here, even though it does not match the expected type"""
    return Model(
        pid=uuid.uuid4(),
        model=torch.jit.load(f"./tests/data/{MODEL_PATH}"), # HACK this is a hack, as the Model.model is expected to be an InferenceSession not a torch.nn.Module
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


def test_non_empty_registry():
    assert len(model_metric_registry._functions) > 0


def test_data_metric_registry_contains_evaluator(
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TabularClassificationModel,
):
    metric_name = "local_lipschitz_estimate"
    metric_func = model_metric_registry.get_functions()[metric_name]

    measures = metric_func(
        data_shape, ref_model, test_dataset, functional_model
    )
    assert len(measures) > 0
    save_measures(metric_name, measures)