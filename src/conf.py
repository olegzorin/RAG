import logging
import os
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

def get_log_level() -> int:
    return int(properties.get(f'ppc.ragagent.logLevel', PropertyTuple(data=logging.WARN, meta=None)).data)
