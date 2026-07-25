from __future__ import annotations

import argparse
import json
from pathlib import Path

from prepare_lammps_ghost_target_input import (
    DEFAULT_ELEMENT_ORDER,
    DEFAULT_MASSES,
    PDBAtom,
    parse_element_order,
    parse_pdb,
    wrap_position,
    write_lammps_input,
)


def write_lammps_data(
    *,
    path: Path,
    atoms: list[PDBAtom],
    cell: tuple[float, float, float],
    element_order: tuple[str, ...],
) -> dict[str, object]:
    base_types = {element: i + 1 for i, element in enumerate(element_order)}

    for atom in atoms:
        if atom.element not in base_types:
            raise ValueError(
                f"Atom index {atom.index} has element {atom.element}, not in {element_order}."
            )

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("LAMMPS data generated from PDB for normal MatterTune-MatterSim MD\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{len(element_order)} atom types\n\n")
        handle.write(f"0.0 {cell[0]:.10f} xlo xhi\n")
        handle.write(f"0.0 {cell[1]:.10f} ylo yhi\n")
        handle.write(f"0.0 {cell[2]:.10f} zlo zhi\n\n")
        handle.write("Masses\n\n")

        for atom_type, element in enumerate(element_order, start=1):
            handle.write(f"{atom_type} {DEFAULT_MASSES[element]:.12g} # {element}\n")

        handle.write("\nAtoms # atomic\n\n")

        for atom in atoms:
            x = wrap_position(atom.x, cell[0])
            y = wrap_position(atom.y, cell[1])
            z = wrap_position(atom.z, cell[2])
            handle.write(
                f"{atom.index + 1} {base_types[atom.element]} "
                f"{x:.10f} {y:.10f} {z:.10f} "
                f"# pdb_index={atom.index} pdb_serial={atom.serial} "
                f"name={atom.name} residue={atom.residue}{atom.residue_id}\n"
            )

    return {
        "natoms": len(atoms),
        "cell": cell,
        "element_order": element_order,
        "base_types": base_types,
        "pair_coeff_elements": element_order,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LAMMPS atomic data and input for normal MatterTune-MatterSim "
            "ML-IAP MD from an electrolyte PDB."
        )
    )

    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--element-order", type=parse_element_order, default=DEFAULT_ELEMENT_ORDER)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--thermo-interval", type=int, default=100)
    parser.add_argument("--dump-interval", type=int, default=100)
    parser.add_argument("--xtc-dump-interval", type=int, default=500)
    parser.add_argument("--xtc-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-velocities", dest="init_velocities", action="store_true", default=True)
    parser.add_argument("--no-init-velocities", dest="init_velocities", action="store_false")
    parser.add_argument("--final-data", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    atoms, cell = parse_pdb(args.pdb)

    metadata = write_lammps_data(
        path=args.data,
        atoms=atoms,
        cell=cell,
        element_order=args.element_order,
    )

    write_lammps_input(
        path=args.input,
        data_path=args.data.resolve(),
        model_path=args.model.resolve(),
        pair_coeff_elements=list(metadata["pair_coeff_elements"]),
        temperature=args.temperature,
        timestep_fs=args.timestep_fs,
        friction_fs_inv=args.friction_fs_inv,
        warmup_steps=args.warmup_steps,
        steps=args.steps,
        thermo_interval=args.thermo_interval,
        dump_interval=args.dump_interval,
        xtc_dump_interval=args.xtc_dump_interval,
        xtc_path=args.xtc_path.resolve(),
        seed=args.seed,
        init_velocities=args.init_velocities,
        final_data_path=args.final_data.resolve(),
        dump_path=args.dump.resolve(),
    )

    payload = {
        "pdb": str(args.pdb),
        "data": str(args.data),
        "input": str(args.input),
        "model": str(args.model),
        "temperature": args.temperature,
        "timestep_fs": args.timestep_fs,
        "friction_fs_inv": args.friction_fs_inv,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "thermo_interval": args.thermo_interval,
        "dump_interval": args.dump_interval,
        "xtc_dump_interval": args.xtc_dump_interval,
        "xtc_path": str(args.xtc_path),
        "seed": args.seed,
        "init_velocities": args.init_velocities,
        **metadata,
    }

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {args.data}")
    print(f"wrote {args.input}")
    print(f"wrote {args.metadata}")
    print(f"xtc_dump_interval={args.xtc_dump_interval}")
    print(f"xtc_path={args.xtc_path}")
    print(f"pair_coeff * * {' '.join(metadata['pair_coeff_elements'])}")


if __name__ == "__main__":
    main()