from __future__ import annotations

import contextlib
import subprocess
import threading
import time

import torch.distributed as dist
import os
import random
import numpy as np
import torch
from ase import Atoms
import ase.geometry


@contextlib.contextmanager
def optional_import_error_message(pip_package_name: str, /):
    try:
        yield
    except ImportError as e:
        raise ImportError(
            f"The `{pip_package_name}` package is not installed. Please install it by running "
            f"`pip install {pip_package_name}`."
        ) from e


def is_rank_zero():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def set_global_random_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def write_to_npz(
    atoms:Atoms|list[Atoms],
    filename:str
):
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    data_dict = {}
    for i, atoms in enumerate(atoms):
        pos = np.array(atoms.get_positions())
        atomic_numbers = np.array(atoms.get_atomic_numbers())
        cell = np.array(atoms.get_cell(complete=True))
        pbc = np.array(atoms.pbc, dtype=np.int32)
        info = atoms.info
        for key, value in info.items():
            try:
                data_dict[f"{key}_{i}"] = np.array(value)
            except:
                print(f"Failed to store {key} in npz file")
        data_dict[f"pos_{i}"] = pos
        data_dict[f"atomic_numbers_{i}"] = atomic_numbers
        data_dict[f"cell_{i}"] = cell
        data_dict[f"pbc_{i}"] = pbc
    np.savez(filename, **data_dict)

def load_from_npz(filename:str)->list[Atoms]:
    data = np.load(filename)
    keys = list(data.keys())
    num_structs = max([int(key.split("_")[-1]) for key in keys if key.startswith("pos")]) + 1
    atoms_list = []
    for i in range(num_structs):
        pos = data[f"pos_{i}"]
        atomic_numbers = data[f"atomic_numbers_{i}"]
        cell = data[f"cell_{i}"]
        pbc = data[f"pbc_{i}"]
        # convert pbc to tuple[bool, bool, bool]
        pbc = tuple(bool(p) for p in pbc)
        atoms = Atoms(numbers=atomic_numbers, positions=pos, cell=cell, pbc=pbc)
        info = {}
        for key in keys:
            if key.endswith(f"_{i}") and "pos" not in key and "atomic_numbers" not in key and "cell" not in key and "pbc" not in key:
                info["_".join(key.split("_")[:-1])] = data[key]
        atoms.info = info
        atoms_list.append(atoms)
    return atoms_list

def neighbor_list_and_relative_vec(
    method: str,
    *,
    pos: np.ndarray,
    cell: np.ndarray,
    r_max: float,
    self_interaction=False,
    pbc: bool | tuple[bool, bool, bool] | np.ndarray = True,
    k_neighbors: int | None = None,
):
    """
    Copied from Nequip Repo
    Compute connectivity of atoms within a cutoff radius using various implementations.
    https://github.com/mir-group/nequip/blob/main/nequip/data/AtomicData.py
    """
    with optional_import_error_message("ase"):
        import ase.neighborlist as ase_nl
    
    with optional_import_error_message("pymatgen"):
        from pymatgen.optimization.neighbors import find_points_in_spheres
        
    with optional_import_error_message("vesin"):
        from vesin import NeighborList as vesin_nl
    
    with optional_import_error_message("matscipy"):
        import matscipy.neighbours
    
    if isinstance(pbc, bool):
        pbc = (pbc, pbc, pbc)
    elif isinstance (pbc, np.ndarray):
        pbc = tuple(pbc)
    cell = ase.geometry.complete_cell(cell)
    if k_neighbors is not None:
        quantity = "ijd"
    else:
        quantity = "ij"
    if method == "vesin":
        # use same mixed pbc logic as
        # https://github.com/Luthaf/vesin/blob/main/python/vesin/src/vesin/_ase.py
        if pbc[0] and pbc[1] and pbc[2]:
            periodic = True
        elif not pbc[0] and not pbc[1] and not pbc[2]:
            periodic = False
        else:
            raise ValueError(
                "different periodic boundary conditions on different axes are not supported by vesin neighborlist, use ASE or matscipy"
            )

        results= vesin_nl(
            cutoff=float(r_max), full_list=True
        ).compute(points=pos, box=cell, periodic=periodic, quantities=quantity)
        first_idex = results[0]
        second_idex = results[1]
        if k_neighbors is not None:
            distances = results[2]

    elif method == "matscipy":
        results = matscipy.neighbours.neighbour_list(
            quantity,
            pbc=pbc,
            cell=cell,
            positions=pos,
            cutoff=float(r_max),
        )
        first_idex = results[0]
        second_idex = results[1]
        if k_neighbors is not None:
            distances = results[2]
    elif method == "ase":
        results = ase_nl.primitive_neighbor_list(
            quantity,
            pbc,
            cell,
            pos,
            cutoff=float(r_max),
            self_interaction=self_interaction,  # we want edges from atom to itself in different periodic images!
            use_scaled_positions=False,
        )
        first_idex = results[0]
        second_idex = results[1]
        if k_neighbors is not None:
            distances = results[2]
    elif method == "pymatgen":
        first_idex, second_idex, shifts, distances = find_points_in_spheres(
            pos, pos, r=r_max, pbc=np.array(pbc, dtype=int), lattice=np.array(cell), tol=1e-8
        )
    else:
        raise ValueError(f"Unknown method {method}")

    # Eliminate true self-edges that don't cross periodic boundaries
    if not self_interaction:
        bad_edge = first_idex == second_idex
        keep_edge = ~bad_edge
        first_idex = first_idex[keep_edge]
        second_idex = second_idex[keep_edge]
        if k_neighbors is not None:
            distances = distances[keep_edge]   # type: ignore
        
    # If k_neighbors is not None, we need to sort the edges by distance
    if k_neighbors is not None:
        perm = np.lexsort((distances, first_idex)) # type: ignore
        fi_sorted  = first_idex[perm] 
        uniq, counts = np.unique(fi_sorted, return_counts=True)
        cumsum = np.cumsum(counts)
        rank_sorted = np.arange(len(fi_sorted)) - np.repeat(cumsum - counts, counts)
        keep_sorted = rank_sorted < k_neighbors
        keep_mask = np.zeros_like(keep_sorted, dtype=bool)
        keep_mask[perm] = keep_sorted
        
        first_idex = first_idex[keep_mask]
        second_idex = second_idex[keep_mask]

    # Build output:
    edge_indices = np.vstack((first_idex, second_idex)).astype(np.int32)
    return edge_indices

def rdf_compute(atoms: Atoms, r_max, n_bins, elements=None, indices=None):
    """
    Compute RDF for a given ASE Atoms object.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
    r_max : float
        Maximum radius (Å)
    n_bins : int
        Number of bins
    elements : tuple(str,str) or None
        If not None, compute partial RDF for element pair (A,B)
    indices : array-like or None
        Subset of atom indices to consider as the "reference" atoms.
        If None, use all atoms.
    """
    with optional_import_error_message("pymatgen"):
        from pymatgen.optimization.neighbors import find_points_in_spheres
    
    scaled_pos = atoms.get_scaled_positions()
    atoms.set_scaled_positions(np.mod(scaled_pos, 1))

    num_atoms = len(atoms)
    volume = atoms.get_volume()
    density = num_atoms / volume

    pos = np.array(atoms.get_positions())
    cell = np.array(atoms.get_cell(complete=True))
    pbc = np.array(atoms.pbc, dtype=int)
    send_indices, receive_indices, _, distances = find_points_in_spheres(
        pos, pos, r=r_max, pbc=pbc, lattice=cell, tol=1e-8
    )
    exclude_self = np.where(send_indices != receive_indices)[0]
    send_indices = send_indices[exclude_self]
    receive_indices = receive_indices[exclude_self]
    distances = distances[exclude_self]
    
    if elements is not None and len(elements) == 2:
        species = np.array(atoms.get_chemical_symbols())
        element_mask = np.logical_and(
                species[send_indices] == elements[0],
                species[receive_indices] == elements[1],
            )
    else:
        element_mask = np.ones_like(send_indices, dtype=bool)
    
    if indices is not None:
        include_mask = np.isin(send_indices, indices)
    else:
        include_mask = np.ones_like(send_indices, dtype=bool)
        
    mask = np.logical_and(element_mask, include_mask)
    send_indices = send_indices[mask]
    distances = distances[mask]
    num_atoms = len(np.unique(send_indices))

    hist, bin_edges = np.histogram(distances, range=(0, r_max), bins=n_bins)
    rdf_x = 0.5 * (bin_edges[1:] + bin_edges[:-1]) + 0.5 * r_max / n_bins
    bin_volume = (4 / 3) * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)
    num_species = len(set(elements)) if elements is not None else 1
    rdf_y = hist / (bin_volume * density * num_atoms) / num_species

    return rdf_x, rdf_y


class NvidiaSmiMonitor:
    def __init__(self, device_id=0, interval=1.0, wait_for_start: float = 0.0):
        self.device_id = device_id
        self.interval = interval
        self.wait_for_start = wait_for_start

        self.memory_records = []

        self._stop_event = threading.Event()
        self._thread = None

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                cmd = [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                    "-i", str(self.device_id)
                ]
                output = subprocess.check_output(cmd)
                gpu_mem_str = output.decode("utf-8").strip()
                gpu_mem = int(gpu_mem_str)

                # 存储记录
                self.memory_records.append(gpu_mem)

            except subprocess.CalledProcessError as e:
                print(f"nvidia-smi error: {e}")
                self.memory_records.append(None)

            time.sleep(self.interval)

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        time.sleep(self.wait_for_start)
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> np.ndarray:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        memory_records = np.array(self.memory_records)
        return memory_records
    
    
import re
from collections.abc import Sequence

PatternLike = str | re.Pattern

def param_matches_group(name: str, group_by: PatternLike | Sequence[PatternLike]) -> bool:
    """
    Return True iff `param_name` matches ANY of the given group_by patterns.

    Parameters
    ----------
    param_name : str
        Full parameter name from `named_parameters()`, e.g. "backbone.blocks.0.attn.q_proj.weight".
    group_by : str | re.Pattern | Sequence[str|re.Pattern]
        Regex pattern(s). Can be raw strings or precompiled `re.Pattern`s.

    Examples
    --------
    >>> param_matches_group("backbone.fc.weight", r"^backbone")
    True
    >>> param_matches_group("head.out.bias", [r"^backbone", r"\.out\."])
    True
    >>> param_matches_group("embed.weight", [r"^backbone", r"^head"])
    False
    """
    patterns: list[PatternLike]
    if isinstance(group_by, (str, re.Pattern)):
        patterns = [group_by]
    else:
        patterns = list(group_by)

    for pat in patterns:
        rex = re.compile(pat) if isinstance(pat, str) else pat
        if rex.search(name):
            return True
    return False