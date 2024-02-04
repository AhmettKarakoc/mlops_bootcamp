"""This is CLI Application for Template Data Science Project"""

# import logging

import typer
from heps_ds_utils import BigQueryOperations, execute_from_bq_file

from src.config.definitions import ROOT_DIR
from src.config.prepare_conf import get_env_configs
from src.utils.cli_app_utils import env_callback
from src.utils.custom_logger import cloud_logger

app = typer.Typer()
project_configs = {}


@app.command()
def some_function():
    """
    This is a sample function
    """
    print("Hello World")


@app.command()
def execute_bq_query(bq_file: str, dependent_queries: bool = False):
    """This is a command to execute a BQ command file.

    Args: Path of the BQ file to execute relative to bigquery directory.
    """
    QUERY_DIR = ROOT_DIR.joinpath("src/data/bigquery")
    bq_ds = BigQueryOperations(gcp_key_path=project_configs["gcp_key"])
    execute_from_bq_file(
        bq_ds,
        bq_file_path=QUERY_DIR.joinpath(bq_file),
        verbose=True,
        config=project_configs,
        dependent_queries=dependent_queries,
    )


@app.callback()
def ds_app(
    env: str = typer.Option(
        ..., "--env", "-e", help="Environment", callback=env_callback
    )
):
    """Application Callback

    Args:
        env (str, optional): _description_. Defaults to typer.Option( ..., "--env", "-e", help="Environment", callback=env_callback ).
    """

    project_configs.update(get_env_configs(env=env))
    _ = cloud_logger(config=project_configs)


if __name__ == "__main__":
    app()
