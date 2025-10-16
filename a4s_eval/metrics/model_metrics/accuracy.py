from datetime import datetime
import numpy as np
import pandas as pd

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel
from a4s_eval.utils.logging import get_logger


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:
    features = dataset.data[[f.name for f in datashape.features]]

    y = dataset.data[datashape.target.name]
    y_pred = functional_model.predict(features.to_numpy())

    accuracy_value = np.mean(y == y_pred)
    current_time = datetime.now()
    return [Measure(name="accuracy", score=accuracy_value, time=current_time)]
