"""Neuromorphic-hardware layer: device non-idealities (training + evaluation) and platform cost models."""

from .cost import PLATFORMS, EnergyTable, estimate_cost, load_energy_table
from .nonideal import DeviceConfig

__all__ = ["PLATFORMS", "EnergyTable", "estimate_cost", "load_energy_table", "DeviceConfig"]
