import logging
import os
import sys
import warnings
from pathlib import Path

from jproperties import Properties, PropertyTuple

ppc_home = os.getenv('PPC_HOME', '/opt/greenxserver')

# Properties

properties = Properties()
props_path = Path(ppc_home, 'config', 'properties', 'ragagent.properties')
with open(props_path, 'rb') as props_file:
    properties.load(props_file)


def get_property(key: str, default: str = None) -> str:
    return properties.get(f'ppc.ragagent.{key}', PropertyTuple(data=default, meta=None)).data


def _resolve_path(property_key: str, default_path: str) -> Path:
    return Path(ppc_home, 'ragagent', get_property(property_key, default_path))

docs_cache_dir = _resolve_path("docs.cacheDir", "documents").as_posix()
model_cache_dir = _resolve_path('models.cacheDir', 'caches').as_posix()
model_source_dir = _resolve_path('models.sourceDir', 'models').as_posix()

# Logging

def _get_log_level() -> int:
    return int(properties.get(f'ppc.ragagent.logLevel', PropertyTuple(data=logging.WARN, meta=None)).data)


def set_logging():
    logging.basicConfig(
        stream=sys.stderr,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=_get_log_level(),
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True
    )

    warnings.filterwarnings(
        action="ignore",
        category=DeprecationWarning
    )
    warnings.filterwarnings(
        action="ignore",
        category=FutureWarning
    )
    warnings.filterwarnings(
        action="error",
        category=UserWarning
    )
