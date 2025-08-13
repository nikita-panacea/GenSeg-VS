"""Shim to register dataset_mode 'vscsv' with the loader.
This exposes class VscsvDataset defined in vs_dataset_csv.py
under module name data.vscsv_dataset, as expected by the framework.
"""
from .vs_dataset_csv import VscsvDataset


