# -*- coding: utf-8 -*-
import click

from swes.config import Config
from swes.driver import Driver


@click.command()
@click.option("-c", "--config-file", type=str, help="Path to the configuration file.")
def main(config_file: str) -> None:
    config = Config.from_yaml(config_file)
    driver = Driver(config)
    driver.run()


if __name__ == "__main__":
    main()
