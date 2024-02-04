""" This module implemented to handle the data transfer between BigQuery and Python Applications. """
import time

import pandas as pd
from heps_ds_utils import BigQueryOperations


def read_master_data(config: dict, table_name: str, **kwargs):
    """
    This function to read data from BigQuery.

    Args:
        config (dict): project configs
        table_name (str): table name to read from
        **kwargs:
            - query (str): query to run
    Returns:
        pd.DataFrame: Returns a dataframe with the data from the table.
    """
    print(f"Retrieving Data From: {table_name} ")
    bq_ds = BigQueryOperations(gcp_key_path=config["gcp_key"])
    if not kwargs.get("query"):
        QUERY_STRING = f"SELECT * FROM `{table_name}`;"
    else:
        QUERY_STRING = kwargs.get("query")
    data = bq_ds.execute_query(QUERY_STRING, return_type="dataframe")
    return data


def send_results_to_bq(
    config: dict, data: pd.DataFrame, table_name: str, overwrite: bool
):
    """
    This function to send data to BigQuery.

    Args:
        config (dict): project configs
        data (pd.DataFrame): data to send to bq
        table_name (str): table name to send data to
        overwrite (bool): overwrite table or not. True to overwrite, False to append
    """
    bq_ds = BigQueryOperations(gcp_key_path=config["gcp_key"])

    start_result_write_timer = time.time()
    bq_ds.load_data_to_table(
        dataset=config["db_prefix"],
        table_name=table_name,
        data_frame=data,
        overwrite=overwrite,
    )
    end_result_write_timer = time.time()
    print(
        f"Results sent to BQ in {end_result_write_timer - start_result_write_timer:.2f} seconds"
    )
