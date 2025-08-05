# ruff: noqa: F401
from a4s_eval.celery_app import celery_app
from a4s_eval.celery_tasks import poll_and_run_evaluation
from a4s_eval.tasks.evaluation_tasks import dataset_evaluation_task

# This file is imported by the Celery worker to register all tasks
