import json
import logging
from pathlib import Path


class BaseConfig(object):

    def __init__(self, config_dict: dict, name: str):
        self.config_dict = config_dict
        self.name = name

    @classmethod
    def from_json_file(cls, config_file: Path):

        with open(config_file, "r", encoding="utf8") as cfile:
            config_dict = json.load(cfile)

        return BaseConfig(config_dict, config_dict["name"])

    def log_configurations(self, logger: logging.Logger):

        logger.info("Used configurations:")
        for key, value in self.__dict__.items():
            if key not in ["config_dict"]:
                logger.info(f"\t{key}: {value}")

    def to_json_file(self, output_file: Path):
        json_data = {}
        for key, value in self.__dict__.items():
            if key not in ["config_dict"]:
                json_data[key] = value

        with open(output_file, "w", encoding="utf8") as f:
            json.dump(json_data, f, indent=4)
