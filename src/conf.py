import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

from jproperties import Properties, PropertyTuple

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_tesseract_version():
    try:
        import subprocess

        output = subprocess.check_output(
            args=['tesseract', '--version'],
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=os.environ
        )

        import re

        if m := re.match(r'tesseract (?P<n1>\d+)\.(?P<n2>\d+)\.(?P<n3>\d+)', output.decode('utf-8')):
            return int(m.group('n1')), int(m.group('n2')), int(m.group('n3'))
    except OSError:
        print('Tesseract not installed')
    return None


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


DOCS_CACHE_DIR = _resolve_path("docs.cacheDir", "documents").as_posix()
MODEL_CACHE_DIR = _resolve_path('models.cacheDir', 'caches').as_posix()
MODEL_SOURCE_DIR = _resolve_path('models.sourceDir', 'models').as_posix()


# Logging

def _get_log_level() -> int:
    return int(properties.get(f'ppc.ragagent.logLevel', PropertyTuple(data=logging.WARN, meta=None)).data)


def set_logging(log_level: Optional[int] = None):
    logging.basicConfig(
        stream=sys.stderr,
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=log_level or _get_log_level(),
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
