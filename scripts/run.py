# -*- coding: utf-8 -*-
#
# py-swes: A Python solver for the Shallow Water Equations over a Sphere (SWES)
# Copyright (C) 2016-2023, ETH Zurich
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
