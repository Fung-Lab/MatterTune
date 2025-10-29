import torch
import torch.nn as nn
from typing_extensions import override

from ...util import optional_import_error_message


class GeneralNeighborListTransform(nn.Module):
    """
    A general neighbor list transformation module for SchNet/PaiNN-based models,
    with optional Verlet-skin reuse (set `skin>0` to enable).
    """
    def __init__(
        self,
        cutoff: float,
        fn_type: str = "pymatgen",
        skin: float | None = None,
        mic_eps: float = 1e-8,
        cell_rtol: float = 1e-6,
        cell_atol: float = 1e-8,
    ):
        super().__init__()
        assert fn_type in ["pymatgen", "ase", "vesin", "matscipy"], \
            f"Invalid fn_type specified, expected one of ['pymatgen', 'ase', 'vesin', 'matscipy'], got {fn_type}"
        self.cutoff = float(cutoff)
        self.fn_type = fn_type
        self.skin = float(skin) if isinstance(skin, float) and skin > 0.0 else None

        # MIC & cell compare tolerances
        self._mic_eps = float(mic_eps)
        self._cell_rtol = float(cell_rtol)
        self._cell_atol = float(cell_atol)

        # cache: sample_idx -> dict
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    @override
    def forward(
        self,
        inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        with optional_import_error_message("schnetpack"):
            from schnetpack import properties
            
        sample_idx = int(inputs.get(properties.idx, torch.tensor(0)).item())
        

        Z = inputs[properties.Z]
        R = inputs[properties.R]                  # [N,3]
        cell = inputs[properties.cell].view(3, 3) # [3,3]
        pbc = inputs[properties.pbc]              # [3], bool

        # if no skin: always rebuild
        if not self.skin:
            idx_i, idx_j, offsets = self._build_neighbor_list(
                Z, R, cell, pbc, cutoff=self.cutoff
            )
            inputs[properties.idx_i] = idx_i.detach()
            inputs[properties.idx_j] = idx_j.detach()
            inputs[properties.offsets] = offsets
            return inputs

        # with skin: check whether to rebuild
        need_rebuild = self._need_rebuild(sample_idx, R, cell, pbc)
        if need_rebuild:
            idx_i, idx_j, offsets = self._build_neighbor_list(
                Z, R, cell, pbc, cutoff=self.cutoff + self.skin
            )
            # update cache
            self._cache[sample_idx] = {
                properties.R: R.detach().clone(),
                properties.cell: cell.detach().clone(),
                properties.pbc: pbc.detach().clone(),
                properties.idx_i: idx_i.detach().clone(),
                properties.idx_j: idx_j.detach().clone(),
                properties.offsets: offsets.detach().clone(),
            }
        else:
            # reuse cached neighbor list
            c = self._cache[sample_idx]
            idx_i, idx_j = c[properties.idx_i], c[properties.idx_j]
            # IMPORTANT: recompute offsets to account for atom displacements
            offsets = self._recompute_offsets_mic(R, idx_i, idx_j, cell, pbc)

        Rij = self._pair_vectors(R, idx_i, idx_j, offsets)           # [E,3]
        keep = (Rij * Rij).sum(dim=-1) <= (self.cutoff * self.cutoff + self._mic_eps)
        inputs[properties.idx_i] = idx_i[keep].detach()
        inputs[properties.idx_j] = idx_j[keep].detach()
        inputs[properties.offsets] = offsets[keep]
        return inputs

    def _need_rebuild(
        self,
        sample_idx: int,
        R: torch.Tensor,        # [N,3]
        cell: torch.Tensor,     # [3,3]
        pbc: torch.Tensor,      # [3] bool
    ) -> bool:
        
        if sample_idx not in self._cache:
            return True
        prev = self._cache[sample_idx]
        
        with optional_import_error_message("schnetpack"):
            from schnetpack import properties

        if not self._same_pbc(prev[properties.pbc], pbc):
            return True
        if not self._same_cell(prev[properties.cell], cell):
            return True

        max_sq_disp = self._max_sq_displacement_mic(
            prev[properties.R].to(device=R.device, dtype=R.dtype),
            R,
            cell,
            pbc,
        )
        thresh = (0.5 * self.skin) ** 2 - self._mic_eps # type: ignore[assignment]
        return not bool(max_sq_disp <= thresh)
    
    def _recompute_offsets_mic(
        self, R: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor,
        cell: torch.Tensor, pbc: torch.Tensor
    ) -> torch.Tensor:
        inv_cell = torch.linalg.inv(cell)                     # [3,3]
        Rf = R @ inv_cell.T                                # [N,3]
        dRf = Rf[idx_j] - Rf[idx_i]                        # [E,3]

        if pbc.any():
            pmask = pbc.to(dtype=dRf.dtype, device=dRf.device)  # [3]
            shift = torch.round(dRf) * pmask                    
        else:
            shift = torch.zeros_like(dRf)

        # Rij = (Rj - Ri) + offsets, offsets = (-shift) @ cell^T
        offsets = (-shift) @ cell                         # [E,3]
        return offsets

    @staticmethod
    def _to_frac(R: torch.Tensor, cell: torch.Tensor) -> torch.Tensor:
        # R: [N,3], cell: [3,3]
        inv_cell = torch.inverse(cell)
        return R @ inv_cell.T

    def _max_sq_displacement_mic(
        self,
        R_prev: torch.Tensor,  # [N,3]
        R: torch.Tensor,       # [N,3]
        cell: torch.Tensor,    # [3,3]
        pbc: torch.Tensor,     # [3] bool
    ) -> torch.Tensor:
        Rf_prev = self._to_frac(R_prev, cell)
        Rf      = self._to_frac(R,      cell)
        dRf = Rf - Rf_prev
        if pbc.any():
            pbc_mask = pbc.to(dtype=dRf.dtype, device=dRf.device)
            wrapped = dRf - torch.round(dRf)      
            dRf = wrapped * pbc_mask + dRf * (1.0 - pbc_mask)
        dR = dRf @ cell.T
        return (dR * dR).sum(dim=-1).max()

    def _same_pbc(self, p0: torch.Tensor, p1: torch.Tensor) -> bool:
        return bool(torch.equal(p0, p1))

    def _same_cell(self, c0: torch.Tensor, c1: torch.Tensor) -> bool:
        return bool(torch.allclose(c0, c1, rtol=self._cell_rtol, atol=self._cell_atol))

    @staticmethod
    def _pair_vectors(
        R: torch.Tensor, idx_i: torch.Tensor, idx_j: torch.Tensor, offsets: torch.Tensor
    ) -> torch.Tensor:
        
        return R[idx_j] - R[idx_i] + offsets


    def _build_neighbor_list(
        self,
        Z: torch.Tensor,
        positions: torch.Tensor,   # [N,3]
        cell: torch.Tensor,        # [3,3]
        pbc: torch.Tensor,         # [3]
        cutoff: float,             # scalar
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        pos_np = positions.detach().cpu().numpy().reshape(-1, 3)
        cell_np = cell.detach().cpu().numpy().reshape(3, 3)
        pbc_np_bool = pbc.detach().cpu().numpy().astype(bool).reshape(-1)
        pbc_np_int = pbc_np_bool.astype(int).reshape(-1)

        device = positions.device
        dtype = positions.dtype

        match self.fn_type:
            case "pymatgen":
                with optional_import_error_message("pymatgen"):
                    from pymatgen.optimization.neighbors import find_points_in_spheres

                idx_i, idx_j, offsets, distances = find_points_in_spheres(
                    pos_np,
                    pos_np,
                    r=float(cutoff),
                    pbc=pbc_np_int,
                    lattice=cell_np,
                    tol=1e-8,
                )
                # remove self-interactions
                mask = idx_i != idx_j
                idx_i = torch.from_numpy(idx_i[mask]).to(device)
                idx_j = torch.from_numpy(idx_j[mask]).to(device)
                offsets_frac = torch.from_numpy(offsets[mask]).to(dtype=dtype, device=device)
                offsets_cart = offsets_frac @ cell.to(device)   # [E,3]
                return idx_i, idx_j, offsets_cart

            case "ase":
                with optional_import_error_message("ase"):
                    import ase.neighborlist as nl
                idx_i, idx_j, S = nl.primitive_neighbor_list(
                    "ijS",
                    pbc=pbc_np_bool,
                    cell=cell_np,
                    positions=pos_np,
                    cutoff=float(cutoff),
                    self_interaction=False,
                )
                idx_i = torch.from_numpy(idx_i).to(device)
                idx_j = torch.from_numpy(idx_j).to(device)
                S = torch.from_numpy(S).to(dtype=dtype, device=device)
                offsets_cart = S @ cell.to(device)
                return idx_i, idx_j, offsets_cart

            case "vesin":
                with optional_import_error_message("vesin"):
                    from vesin import NeighborList as vesin_nl
                if pbc_np_bool.all():
                    periodic = True
                elif (~pbc_np_bool).all():
                    periodic = False
                else:
                    raise ValueError("vesin neighbor list does not support mixed PBC settings.")
                results = vesin_nl(
                    cutoff=float(cutoff), full_list=True
                ).compute(
                    points=pos_np, box=cell_np, periodic=periodic, quantities="ijS"
                )
                idx_i = torch.from_numpy(results[0]).to(device).to(torch.long)
                idx_j = torch.from_numpy(results[1]).to(device).to(torch.long)
                S = torch.from_numpy(results[2]).to(dtype=dtype, device=device)
                offsets_cart = S @ cell.to(device)
                return idx_i, idx_j, offsets_cart

            case "matscipy":
                with optional_import_error_message("matscipy"):
                    import matscipy.neighbours as mat_nl
                ijS = mat_nl.neighbour_list(
                    "ijS",
                    pbc=pbc_np_bool,
                    cell=cell_np,
                    positions=pos_np,
                    cutoff=float(cutoff),
                )
                idx_i = torch.from_numpy(ijS[0]).to(device)
                idx_j = torch.from_numpy(ijS[1]).to(device)
                S = torch.from_numpy(ijS[2]).to(dtype=dtype, device=device)
                offsets_cart = S @ cell.to(device)
                return idx_i, idx_j, offsets_cart

            case _:
                raise ValueError(f"Invalid fn_type specified, got {self.fn_type}")

    def reset_skin_cache(self):
        self._cache.clear()
