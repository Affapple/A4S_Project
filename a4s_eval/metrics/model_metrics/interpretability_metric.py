from datetime import datetime
import random
from captum.attr import Lime
import numpy as np
from quantus import LocalLipschitzEstimate
import torch

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.functional_model import TabularClassificationModel


def convert_dtype(np_dtype):
    """
    Convert numpy dtype to torch dtype
    """
    if np_dtype == np.float32:
        return torch.float32
    elif np_dtype == np.float64:
        return torch.float32
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    else:
        raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
    
def tensorize(function):
    """
    Convert all input and output arrays to tensors
    Necessary for compatibility with Captum and improves readability
    """
    def helper(*args, **kwargs):
        args = list(args)
        for key in kwargs:
            if isinstance(kwargs[key], np.ndarray):
                kwargs[key] = torch.tensor(kwargs[key]).to(convert_dtype(kwargs[key].dtype))
        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                args[i] = torch.tensor(args[i]).to(convert_dtype(args[i].dtype))
        
        output = function(*args, **kwargs)
        if isinstance(output, np.ndarray):
            return torch.tensor(output).to(convert_dtype(output.dtype))
        return output
    return helper      


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
        interpretability_function = lambda *args, **kwargs: \
            Lime(tensorize(functional_model.predict_proba)).attribute(
                inputs=kwargs["inputs"], target=kwargs["targets"]
            )
            # Lime(tensorize(functional_model.predict_proba)).attribute(
            #     inputs=kwargs["inputs"][0].reshape(1, -1), target=kwargs["targets"][0].reshape(1)
            # )

    explain_func = lambda *args, **kwargs: \
        tensorize(interpretability_function)(*args, **kwargs).detach().cpu().numpy()


    ## Prepare data
    if dataset.data is None:
        raise AssertionError("Dataset data not defined!")
    if dataset.shape.target is None:
        raise AssertionError("Dataset target not defined!")

    features = [feature.name for feature in dataset.shape.features]
    target = dataset.shape.target.name
    
    # Subsample the dataset for performance (LIME + Lipschitz is expensive)
    # We use a small number of examples to estimate the metric
    N_EXAMPLES = 5
    if len(dataset.data) > N_EXAMPLES:
        # Use random state for reproducibility
        sampled_data = dataset.data.sample(n=N_EXAMPLES, random_state=42)
    else:
        sampled_data = dataset.data

    X_batch = sampled_data[features]
    y_batch = sampled_data[target]

    ## Evaluate 
    metric = LocalLipschitzEstimate(
        nr_samples=5,
        normalise=True,
    )

    scores = metric(
        model=model.model,
        x_batch=X_batch.to_numpy(),
        y_batch=y_batch.to_numpy(),
        batch_size=N_EXAMPLES,
        explain_func=explain_func
    )

    print("Local Lipschitz Estimates:", scores)
    current_time = datetime.now()

    mean = sum(scores) / len(scores)
    return [Measure(name="local_lipschitz_estimate", score=mean, time=current_time)]