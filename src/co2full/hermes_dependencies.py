"""HermesSim model and preprocessing imports."""

from co2full.runtime import asset_root, tool_path, require_server
import argparse
from pathlib import Path
import sys

ROOT = asset_root()
require_server()
sys.path.insert(0, str(tool_path("hermes_root") / "model"))
sys.path.insert(0, str(tool_path("hermes_root") / "preprocess"))
import numpy as np
import torch
from core.ggnn import GGNN, EmbeddingEncoder
from core.aggrs import AdapMultiSoftmaxAggrV2
from core.graph_factory_base import GraphFactoryBase
import preprocessing_pcode as prep
