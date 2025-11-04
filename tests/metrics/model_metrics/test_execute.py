import uuid

from numpy import isin
import pandas as pd
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric_registry
from a4s_eval.metric_registries.model_metric_registry import ModelMetric
from a4s_eval.service.functional_model import TabularClassificationModel
from a4s_eval.service.model_factory import load_model
import pytest

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Model,
    ModelConfig,
    ModelFramework,
    ModelTask,
)

from tests.save_measures_utils import save_measures
from tests.save_plot import draw_plot


@pytest.fixture
def data_shape() -> DataShape:
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    for record in metadata:
        record["pid"] = uuid.uuid4()

    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(tab_class_test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    data = tab_class_test_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(tab_class_train_data, data_shape: DataShape) -> Dataset:
    data = tab_class_train_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
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
        path="./tests/data/lcld_v2_tabtransformer.pt",
        framework=ModelFramework.TORCH,
        task=ModelTask.CLASSIFICATION,
    )

    model = load_model(model_config)
    if not isinstance(model, TabularClassificationModel):
        raise TypeError
    return model


def test_non_empty_registry():
    assert len(model_metric_registry._functions) > 0


@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_data_metric_registry_contains_evaluator(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: TabularClassificationModel,
):
    measures = evaluator_function[1](
        data_shape, ref_model, test_dataset, functional_model
    )
    save_measures(evaluator_function[0], measures)
    assert len(measures) > 0


@pytest.mark.parametrize("evaluator_function", model_metric_registry)
def test_data_metric_registry_contains_evaluator_by_batch(
    evaluator_function: tuple[str, ModelMetric],
    data_shape: DataShape,
    ref_model: Model,
    test_dataset: Dataset,
    functional_model: FunctionalModel,
):
    measures: list[Measure] = []
    
    # This value should be 10_000, however just for testing purposes (since the test dataset only has 100 entries I've used 100)
    BATCH_SIZE = 100
    for current_batch in range(0, len(test_dataset.data), BATCH_SIZE):
        batch_data = test_dataset.data.iloc[current_batch : max(current_batch + BATCH_SIZE, len(test_dataset.data))]
        batch_dataset = Dataset(
            pid=uuid.uuid4(), shape=test_dataset.shape, data=batch_data
        )
        measures.append(
            evaluator_function[1](
                data_shape, ref_model, batch_dataset, functional_model
            )[0]
        )
    
    save_measures(evaluator_function[0] + "_by_batch", measures)

    draw_plot(
        f"{evaluator_function[0]}_by_batch",
        x=None,
        y="score",
        title=f"{evaluator_function[0]} by batch",
        x_label=f"Batch index ({BATCH_SIZE} samples)",
        y_label=evaluator_function[0],
    )
    assert len(measures) > 0