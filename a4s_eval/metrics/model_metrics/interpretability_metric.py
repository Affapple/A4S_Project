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
    Convert numpy dtype to torch dtype \\
    Forces float64 to float32 to avoid issues
    """
    if np_dtype == np.float32 or np_dtype == np.float64:
        return torch.float32
    elif np_dtype == np.int32:
        return torch.int32
    elif np_dtype == np.int64:
        return torch.int64
    else:
        raise ValueError(f"Unsupported numpy dtype: {np_dtype}")
    
def tensorize(function):
    """
    Convert all input and output arrays to tensors \\
    Necessary for compatibility with Captum. \\
    This was made with multipurpose in mind, so it can be used with any function.
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


@model_metric(name="local_lipschitz_estimate") # type: ignore
def local_lipschitz_estimate(
    datashape: DataShape,
    ref_model: Model,
    dataset: Dataset,
    functional_model: TabularClassificationModel,
    explanation_function = None,
    num_examples: int = 5,
    num_samples: int = 20,
) -> list[Measure]:
    """
    Analyses the interpretability of a model using Local Lipschitz Estimate metric.\\
    A lower Local Lipschitz Estimate indicates that the model's explanations are more stable
    to small perturbations in the input, which is generally desirable for interpretability and robustness

    `num_examples` random examples are sampled from the dataset to compute the metric.\\
    `num_samples` samples are used to estimate the Lipschitz constant around each example.\\
    These are hard coded values based on performance considerations.

    :param DataShape datashape: DataShape of the dataset
    :param Model model: Model to evaluate
    :param Dataset dataset: Dataset to evaluate on
    :param TabularClassificationModel functional_model: Functional model wrapper
    :param Optional[Callable] interpretability_function: Optional interpretability function to use, default: Lime.attribute
    :param Optional[int] num_examples: Number of examples from the dataset to use.
    :param Optional[int] num_samples: Number of samples to derive around each example.

    :return list[Measure]: List containing the Local Lipschitz Estimate measure on the 5 random examples.
    """
    # This will always be None, as I believe its out of the scope of the architecture
    # However, as presented in the initial proposition, lime will be used as default
    if explanation_function is None:
        explanation_function = lambda *args, **kwargs: \
            Lime(tensorize(functional_model.predict_proba)).attribute(
                inputs=kwargs["inputs"], target=kwargs["targets"], n_samples=200
            )
        
    explain_func = lambda *args, **kwargs: \
        tensorize(explanation_function)(*args, **kwargs).detach().cpu().numpy()


    ## Prepare data
    assert dataset.data is not None, "Dataset data not defined!"
    assert datashape.target is not None, "Dataset target not defined!"


    features_names = [feature.name for feature in datashape.features]
    target_name = datashape.target.name

    features = np.column_stack([dataset.data[feature] for feature in features_names])
    targets = np.array(dataset.data[target_name])

    num_existing_examples = features.shape[0]
    
    # Subsample the dataset for performance (LIME + Lipschitz is expensive)
    if num_examples < num_existing_examples:
        sampled_data_idxs = random.sample(range(num_existing_examples), num_examples)
    else:
        sampled_data_idxs = range(num_existing_examples)

    X_batch = features[sampled_data_idxs]
    y_batch = targets[sampled_data_idxs]

    ## Evaluate 
    metric = LocalLipschitzEstimate(
        nr_samples=num_samples,
        normalise=True,
    )

    scores = metric(
        model=ref_model.model,
        x_batch=X_batch,
        y_batch=y_batch,
        batch_size=num_existing_examples,
        explain_func=explain_func
    )
    assert scores is not None and len(scores) != 0, \
        "Something went wrong, Local Lipschitz Estimate returned no scores"
    
    current_time = datetime.now()
    return [Measure(name="local_lipschitz_estimate", score=score, time=current_time) for score in scores ]