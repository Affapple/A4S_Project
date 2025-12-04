from typing import Literal
import uuid
from captum.attr import Lime, Saliency
import pandas as pd
import numpy as np
import pytest

from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metrics.model_metrics.interpretability_metric import tensorize
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

INCOME_MODEL_PATH = "income_model.pt"

def _get_income_dataset() -> pd.DataFrame:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    data = pd.read_csv('tests/data/income/adult.csv')
    data.drop_duplicates(inplace=True)
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    data = data.sample(frac=1).reset_index(drop=True)
    categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                            'relationship', 'race', 'sex', 'native.country', 'income']
    for feature in categorical_features:
        label_encoder = LabelEncoder()
        data[feature] = label_encoder.fit_transform(data[feature])
    X = data.drop('income', axis=1)
    y = data['income']
    continuous_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    X[continuous_features] = scaler.fit_transform(X[continuous_features])
    pd_dataset = pd.DataFrame(X)
    pd_dataset['labels'] = y
    return pd_dataset

@pytest.fixture
def train_dataset():
    return _get_income_dataset()

@pytest.fixture
def income_data_shape() -> DataShape:
    data_shape = {
        # This SHOULD be changed for something more meaninful and correct their types
        "features": [
            Feature(
                pid=uuid.uuid4(),
                name="features",
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
def income_train_dataset(train_dataset: pd.DataFrame, income_data_shape: DataShape) -> Dataset:
    data = train_dataset
    return Dataset(
        pid=uuid.uuid4(),
        shape=income_data_shape,
        data=data,
    )


@pytest.fixture
def income_ref_model(income_train_dataset: Dataset) -> Model:
    return Model(
        pid=uuid.uuid4(),
        model=None,
        dataset=income_train_dataset,
    )


@pytest.fixture
def income_functional_model() -> TabularClassificationModel:
    model_config = ModelConfig(
        path=f"tests/data/{INCOME_MODEL_PATH}",
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
def income_test_dataset(income_data_shape: DataShape) -> dict[str, Dataset]:
    # load previously created datasets
    import pickle
    metric_testing_dataset: dict[str, dict[str, np.ndarray]] = pickle.load(open("./tests/data/income_metric_testing_dataset.pkl", "rb"))

    pd_adv_examples = {
        "features": metric_testing_dataset["adv_examples"]["x"],
        "labels": metric_testing_dataset["adv_examples"]["y"]
    }

    pd_original_adv_examples = {
        "features": metric_testing_dataset["original_adv_example"]["x"],
        "labels": metric_testing_dataset["original_adv_example"]["y"]
    }

    pd_well_classified = {
        "features": metric_testing_dataset["well_classified"]["x"],
        "labels": metric_testing_dataset["well_classified"]["y"]
    }
    pd_wrongly_classified = {
        "features": metric_testing_dataset["wrongly_classified"]["x"],
        "labels": metric_testing_dataset["wrongly_classified"]["y"]
    }

    return {
        "adv_examples": Dataset(
            pid=uuid.uuid4(),
            shape=income_data_shape,
            data=pd_adv_examples
        ),
        "original_adv_examples": Dataset(
            pid=uuid.uuid4(),
            shape=income_data_shape,
            data=pd_original_adv_examples
        ),
        "well_classified": Dataset(
            pid=uuid.uuid4(),
            shape=income_data_shape,
            data=pd_well_classified
        ),
        "wrongly_classified": Dataset(
            pid=uuid.uuid4(),
            shape=income_data_shape,
            data=pd_wrongly_classified
        )
    }

@pytest.mark.parametrize("current_dataset", ["adv_examples", "original_adv_examples", "well_classified", "wrongly_classified"])
def test_interpretability_over_tabular(
    current_dataset: Literal["adv_examples", "original_adv_examples", "well_classified", "wrongly_classified"],
    income_data_shape: DataShape,
    income_ref_model: Model,
    income_test_dataset: dict[str, Dataset],
    income_functional_model: TabularClassificationModel,
):
    dataset_df = income_test_dataset[current_dataset]
    metric_name = "local_lipschitz_estimate"
    metric_func = model_metric_registry.get_functions()[metric_name]

    expl_funs = {
        "lime": lambda *args, **kwargs: \
            Lime(tensorize(income_functional_model.predict_proba)).attribute(
                inputs=kwargs["inputs"], target=kwargs["targets"], n_samples=200
            ),
        "saliency": lambda *args, **kwargs:
            Saliency(tensorize(income_functional_model.predict_proba_grad)).attribute(
                inputs=kwargs["inputs"], target=kwargs["targets"]
            )
    }

    for fun_name, fun in expl_funs.items():
        measures = metric_func(
            income_data_shape, income_ref_model, dataset_df, income_functional_model,
            explanation_function = fun # type: ignore
        )
        assert len(measures) > 0

        measure_name = f"income_{metric_name}-{current_dataset}-{fun_name}"
        save_measures(measure_name, measures)