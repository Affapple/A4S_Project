from datetime import datetime
import random
from captum.attr import Saliency, Lime
import numpy as np
from quantus import LocalLipschitzEstimate
import torch


def measure(
    model,
    images,
    labels
):
    model.eval()
    # Create the Saliency explainer once for efficiency.
    # explainer = Saliency(model)
    explainer = Lime(model) # To get log_softmax

    def explain_func(
        model,  # Unused, but passed by quantus. The model from the outer scope is used.
        inputs,
        targets
    ):
        X = torch.from_numpy(inputs).float()
        y = torch.from_numpy(targets).long()
        attributions = explainer.attribute(inputs=X, target=y, n_samples=200)
        return attributions.detach().cpu().numpy()

    # Subsample the dataset for performance (Saliency + Lipschitz is expensive)
    N_EXAMPLES = 1
    idxs = list(range(images.shape[0]))
    if images.shape[0] > N_EXAMPLES:
        sampled_data = random.sample(idxs, N_EXAMPLES)
    else:
        sampled_data = idxs

    X_batch: torch.Tensor = images[sampled_data, :]
    y_batch = labels[sampled_data]

    ## Evaluate
    metric = LocalLipschitzEstimate(
        nr_samples=200,
        normalise=True,
    )

    scores = metric(
        model=model,
        x_batch=X_batch.numpy(),
        y_batch=y_batch.numpy(),
        batch_size=N_EXAMPLES,
        explain_func=explain_func
    )

    print("Local Lipschitz Estimates:", scores)
    current_time = datetime.now()

    mean = sum(scores) / len(scores)
    return [{"name": "local_lipschitz_estimate", "score": score, "time": current_time} for score in scores]