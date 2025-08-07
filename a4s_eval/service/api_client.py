import uuid
from io import BytesIO
from typing import Any

import pandas as pd
import requests
from pydantic import BaseModel

from a4s_eval.data_model.evaluation import Evaluation
from a4s_eval.data_model.metric import Metric
from a4s_eval.utils.env import API_URL_PREFIX
from a4s_eval.utils.logging import get_logger

logger = get_logger()


class EvaluationStatusUpdateDTO(BaseModel):
    status: str


class MetricDTO(BaseModel):
    name: str
    value: float | str


def fetch_pending_evaluations() -> list[uuid.UUID]:
    """Fetch pending evaluations from the API.

    Returns:
        list[uuid.UUID]: List of evaluation PIDs that are pending
    """
    try:
        logger.debug("=== FETCH_PENDING_EVALUATIONS START ===")
        logger.debug(
            f"1. About to call API: {API_URL_PREFIX}/evaluations?status=pending"
        )

        logger.debug("2. Making API request...")
        resp = requests.get(f"{API_URL_PREFIX}/evaluations?status=pending", timeout=30)
        logger.debug(f"2. API call completed. Status: {resp.status_code}")

        if resp.status_code != 200:
            logger.error(f"3. API returned non-200 status: {resp.status_code}")
            return []

        evaluations = resp.json()
        logger.debug(f"4. Parsed evaluations: {evaluations}")

        claimed_pids = []
        logger.debug(f"5. Processing {len(evaluations)} evaluations...")

        for i, e in enumerate(evaluations):
            logger.debug(f"6.{i + 1} Processing evaluation: {e}")

            if claim_evaluation(e["evaluation_pid"]):
                claimed_pids.append(uuid.UUID(e["evaluation_pid"]))
                logger.debug(
                    f"7.{i + 1} Successfully claimed evaluation: {e['evaluation_pid']}"
                )
            else:
                logger.warning(
                    f"8.{i + 1} Failed to claim evaluation: {e['evaluation_pid']}"
                )

        logger.debug(f"9. Final claimed_pids: {claimed_pids}")
        return claimed_pids

    except Exception as e:
        logger.error(f"ERROR in fetch_pending_evaluations: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return []
    finally:
        logger.debug("=== FETCH_PENDING_EVALUATIONS END ===")


def claim_evaluation(evaluation_pid: uuid.UUID) -> bool:
    logger.debug(f"Claiming evaluation {evaluation_pid}")
    try:
        resp = requests.put(
            f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=processing"
        )
        logger.debug(f"Claim response status: {resp.status_code}")

        if resp.status_code == 200:
            logger.debug(f"Successfully claimed evaluation {evaluation_pid}")
            return True
        else:
            logger.debug(f"Failed to claim evaluation {evaluation_pid}")
            return False
    except Exception as e:
        logger.error(f"Error claiming evaluation {evaluation_pid}: {e}")
        return False


def mark_completed(evaluation_pid: uuid.UUID) -> requests.Response:
    return requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?status=done")


def mark_failed(evaluation_pid: uuid.UUID) -> None:
    payload = EvaluationStatusUpdateDTO(status="failed").model_dump()
    return requests.put(f"{API_URL_PREFIX}/evaluations/{evaluation_pid}", json=payload)


def get_dataset_data(dataset_pid: str) -> pd.DataFrame:
    resp = requests.get(f"{API_URL_PREFIX}/datasets/{dataset_pid}/data", stream=True)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")

    if "parquet" in content_type:
        content_buffer = BytesIO(resp.content)
        content_buffer.seek(0)
        return pd.read_parquet(content_buffer)
    elif "csv" in content_type:
        content_buffer = BytesIO(resp.content)
        content_buffer.seek(0)
        return pd.read_csv(content_buffer)
    else:
        raise ValueError("Unsupported dataset format")


def get_evaluation_request(evaluation_pid: uuid.UUID) -> dict[str, Any]:
    resp = requests.get(
        f"{API_URL_PREFIX}/evaluations/{evaluation_pid}?include=project,dataset,model,datashape"
    )
    resp.raise_for_status()
    return resp.json()


def get_evaluation(
    evaluation_pid: uuid.UUID,
) -> Evaluation:
    return Evaluation.model_validate(get_evaluation_request(evaluation_pid))


def post_metrics(evaluation_pid: uuid.UUID, metrics: list[Metric]) -> requests.Response:
    """Post metrics to the API for a specific evaluation.

    Args:
        evaluation_pid: UUID of the evaluation to post metrics for
        metrics: List of metrics to post

    Returns:
        requests.Response: The API response
    """
    logger.debug(
        f"post_metrics called with {len(metrics)} metrics for evaluation {evaluation_pid}"
    )

    payload = [m.model_dump() for m in metrics]
    logger.debug(f"Payload prepared, size: {len(payload)}")
    if len(payload) > 0:
        logger.debug(f"Sample payload item: {payload[0]}")

    url = f"{API_URL_PREFIX}/evaluations/{evaluation_pid}/metrics"
    logger.debug(f"Posting to URL: {url}")

    response = requests.post(url, json=payload)
    logger.debug(f"Response status: {response.status_code}")
    logger.debug(f"Response headers: {dict(response.headers)}")
    logger.debug(f"Response content: {response.text}")

    if response.status_code != 201:
        logger.error(f"ERROR: Expected status 201, got {response.status_code}")
        logger.error(f"Response content: {response.content}")
        raise ValueError(response.content)

    return response
