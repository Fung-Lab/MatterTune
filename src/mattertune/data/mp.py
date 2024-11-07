from __future__ import annotations

import logging
from typing import Literal

import ase
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from torch.utils.data import Dataset
from typing_extensions import override

from ..registry import data_registry
from .base import DatasetConfigBase

log = logging.getLogger(__name__)


@data_registry.register
class MPDatasetConfig(DatasetConfigBase):
    """Configuration for a dataset stored in the Materials Project database."""

    type: Literal["mp"] = "mp"
    """Discriminator for the MP dataset."""

    api: str
    """Input API key for the Materials Project database."""

    fields: list[str]
    """Fields to retrieve from the Materials Project database."""

    query: dict
    """Query to filter the data from the Materials Project database."""

    @override
    def create_dataset(self):
        return MPDataset(self)


class MPDataset(Dataset[ase.Atoms]):
    def __init__(self, config: MPDatasetConfig):
        super().__init__()
        self.config = config

        from mp_api.client import MPRester

        self.mpr = MPRester(config.api)
        if "material_id" not in config.fields:
            config.fields.append("material_id")
        self.docs = self.mpr.summary.search(fields=config.fields, **config.query)

    @override
    def __getitem__(self, idx: int) -> Atoms:
        doc = self.docs[idx]
        mid = doc.material_id
        structure = self.mpr.get_structure_by_material_id(mid)
        adaptor = AseAtomsAdaptor()
        atoms: Atoms = adaptor.get_atoms(structure)
        atoms.info = dict(doc)
        return atoms
