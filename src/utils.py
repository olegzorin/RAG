import json
import logging
import os
import re
import sys
from pathlib import Path

from jproperties import Properties, PropertyTuple

home_dir = os.getenv('PPC_HOME', '/opt/greenxserver')

properties = Properties()
props_path = Path(home_dir, 'config', 'properties', 'ragagent.properties')
with open(props_path, 'rb') as props_file:
    properties.load(props_file)

def get_property(key: str, default: str = None) -> str:
    return properties.get(f'ppc.ragagent.{key}', PropertyTuple(data=default, meta=None)).data

def resolve_path(property_key: str, default_path: str) -> Path:
    return Path(home_dir, 'ragagent', get_property(property_key, default_path))


def get_logger(name):
    log_level = int(properties.get('ppc.ragagent.logLevel', PropertyTuple(data=logging.WARN, meta=None)).data)
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level
    )
    return logging.getLogger(name)

def clean_text(text: str):
    return re.sub('[\0\\s]+', ' ', text).strip()

def get_non_empty_or_none(json_obj):
    is_empty = json_obj is None or not json_obj or not re.search('[a-zA-Z\\d]', json.dumps(json_obj))
    return None if is_empty else json_obj
