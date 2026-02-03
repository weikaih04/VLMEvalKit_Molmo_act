"""
Embodied Benchmarks for VLMEvalKit

This module contains implementations of 13 embodied/spatial benchmarks:
- BLINK
- Cosmos-Reason 1
- CV-Bench
- EmbSpatial-Bench
- ERQA
- MindCube
- Minimal Videos
- OpenEQA
- RefSpatial-Bench
- RoboSpatial-Home (Pointing + VQA)
- SAT
- VSI-Bench
- Where2Place
"""

import os

# Base path for local benchmark data
# Can be overridden via environment variable
EMBODIED_DATA_ROOT = os.environ.get(
    'EMBODIED_DATA_ROOT',
    '/weka/oe-training-default/jieyuz2/improve_segments/molmo_training/benchmark_data'
)

# Import all benchmark classes
from .cv_bench import CVBenchDataset
from .emb_spatial import EmbSpatialBenchDataset
from .sat import SATDataset
from .mindcube import MindCubeDataset
from .blink import BLINKDataset
from .ref_spatial import RefSpatialBenchDataset
from .robo_spatial import RoboSpatialPointingDataset, RoboSpatialVQADataset
from .where2place import Where2PlaceDataset
from .vsi_bench import VSIBenchDataset
from .cosmos_reason import CosmosReason1Dataset
from .minimal_videos import MinimalVideosDataset
from .open_eqa import OpenEQADataset
from .erqa import ERQADataset
from .mmsi_bench import MMSIBenchDataset
from .point_bench import PointBenchDataset

__all__ = [
    'EMBODIED_DATA_ROOT',
    'CVBenchDataset',
    'EmbSpatialBenchDataset',
    'SATDataset',
    'MindCubeDataset',
    'BLINKDataset',
    'RefSpatialBenchDataset',
    'RoboSpatialPointingDataset',
    'RoboSpatialVQADataset',
    'Where2PlaceDataset',
    'VSIBenchDataset',
    'CosmosReason1Dataset',
    'MinimalVideosDataset',
    'OpenEQADataset',
    'ERQADataset',
    'MMSIBenchDataset',
    'PointBenchDataset',
]
