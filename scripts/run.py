# -*- coding: utf-8 -*-
from swes.config import Config
from swes.driver import Driver


def main():
    config = Config.from_yaml("../config/williamson_1.yml")
    driver = Driver(config)
    driver.run()


if __name__ == "__main__":
    main()
