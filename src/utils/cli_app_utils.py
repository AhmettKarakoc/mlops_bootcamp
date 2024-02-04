import typer


def env_callback(value: str):
    if not (value in ["dev", "qa", "prod"]):
        raise typer.BadParameter(
            "Only one of the following is allowed as environment: dev, qa, prod"
        )
    return value
