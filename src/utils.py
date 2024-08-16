import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from jproperties import Properties, PropertyTuple
from pydantic import Json

properties = Properties()
props_path = Path(os.environ['PPC_HOME'], 'config', 'properties', 'ragagent.properties')

with open(props_path, 'rb') as props_file:
    properties.load(props_file)


def get_property(key: str) -> str:
    return properties.get(key, PropertyTuple(data=None, meta=None)).data


def get_int_property(key: str, default_value: int) -> int:
    return int(properties.get(key, PropertyTuple(data=default_value, meta=None)).data)


def get_float_property(key: str, default_value: float) -> float:
    return float(properties.get(key, PropertyTuple(data=default_value, meta=None)).data)


def get_logger(name):
    logging.basicConfig(
        stream=sys.stderr,
        level=get_int_property('ppc.ragagent.logLevel', logging.WARN)
    )
    return logging.getLogger(name)


def clean_text(text: str):
    return re.sub('[\0\\s]+', ' ', text).strip()


def get_non_empty_or_none(json_obj: Json[Any]) -> Json[Any]:
    is_empty = json_obj is None or not json_obj or not re.search('[a-zA-Z\\d]', json.dumps(json_obj))
    return None if is_empty else json_obj
