from typing import Literal
import uuid
import numpy as np
from captum.attr import Lime, Saliency
from torchvision import datasets, transforms
import torchvision.models as models
import torch
import pandas as pd
import pytest

from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metrics.model_metrics.interpretability_metric import tensorize
from a4s_eval.service.functional_model import PredictClassFn, TabularClassificationModel
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

MNIST_MODEL_PATH = "mnist_cnn.pt"

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
def mnist_train_dataset(train_dataset: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = train_dataset
    return Dataset(
        pid=uuid.uuid4(),
        shape=data_shape,
        data=data,
    )


@pytest.fixture
def mnist_ref_model(mnist_train_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=mnist_train_dataset,
    )


@pytest.fixture
def mnist_functional_model() -> TabularClassificationModel:
    model_config = ModelConfig(
        path=f"./tests/data/{MNIST_MODEL_PATH}",
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
def mnist_testing_examples(data_shape: DataShape) -> dict[str, Dataset]:
    # load previously created datasets
    import pickle
    metric_testing_dataset: dict[str, dict[str, np.ndarray]] = pickle.load(open("./tests/data/metric_testing_dataset.pkl", "rb"))

    pd_adv_examples = {
        "images": metric_testing_dataset["adv_examples"]["x"],
        "labels": metric_testing_dataset["adv_examples"]["y"]
    }

    pd_original_adv_examples = {
        "images": metric_testing_dataset["original_adv_example"]["x"],
        "labels": metric_testing_dataset["original_adv_example"]["y"]
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
        "original_adv_examples": Dataset(
            pid=uuid.uuid4(),
            shape=data_shape,
            data=pd_original_adv_examples
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

@pytest.mark.parametrize("current_dataset", ["adv_examples", "original_adv_examples", "well_classified", "wrongly_classified"])
def test_interpretability_over_mnist(
    current_dataset: Literal["adv_examples", "original_adv_examples" "well_classified", "wrongly_classified"],
    mnist_testing_examples: dict[str, Dataset],
    data_shape: DataShape,
    mnist_ref_model: Model,
    mnist_functional_model: TabularClassificationModel,
):
    dataset_df = mnist_testing_examples[current_dataset]
    metric_name = "local_lipschitz_estimate"
    metric_func = model_metric_registry.get_functions()[metric_name]


    expl_funs = {
        "lime": lambda *args, **kwargs: \
            Lime(tensorize(mnist_functional_model.predict_proba)).attribute(
                inputs=kwargs["inputs"], target=kwargs["targets"], n_samples=200
            ),
        "saliency": lambda *args, **kwargs:
            Saliency(tensorize(mnist_functional_model.predict_proba_grad)).attribute(
                inputs=kwargs["inputs"], target=kwargs["targets"]
            )
    }

    for fun_name, fun in expl_funs.items():
        measures = metric_func(
            data_shape, mnist_ref_model, dataset_df, mnist_functional_model,
            explanation_function = fun # type: ignore
        )
        assert len(measures) > 0

        measure_name = f"{metric_name}-{current_dataset}-{fun_name}"
        save_measures(measure_name, measures)