from heps_ds_utils import LoggingOperations


def cloud_logger(config):
    logger = LoggingOperations(
        gcp_key_path=config["gcp_key"],
        logger_name=config["logger_name"],
        project="{project_name}",
        submodule="not_set",
    )
    logger.set_logger()

    return logger
