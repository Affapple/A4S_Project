from datetime import datetime
from captum.attr import Lime
from quantus import LocalLipschitzEstimate

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel

@model_metric(name="interpretability") # type: ignore
def interpretability(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
    interpretability_function = None
) -> list[Measure]:    
    # This will always be None, as I believe its out of the scope of the architecture
    # However, as presented in the initial proposition, lime will be used as default
    if interpretability_function is None:
        interpretability_function = Lime(
            functional_model.predict_proba, # type: ignore
        )

    # Prepare data
    if dataset.data is None:
        raise AssertionError("Dataset data not defined!")
    if dataset.data.target is None:
        raise AssertionError("Dataset target not defined!")
    
    features = [feature.name for feature in dataset.shape.features]
    target = dataset.shape.target
    X_batch, y_batch = dataset.data[features], dataset.data[target] 

    # Evaluate 
    metric = LocalLipschitzEstimate(
        nr_samples=200,
        normalise=True,
    )

    scores = metric(
        model=model.model,
        x_batch=X_batch.to_numpy(),
        y_batch=y_batch.to_numpy(),
        explain_func=interpretability_function
    )

    print("Local Lipschitz Estimates:", scores)
    current_time = datetime.now()

    mean = sum(scores) / len(scores)
    return [Measure(name="local_lipschitz_estimate", score=mean, time=current_time)]