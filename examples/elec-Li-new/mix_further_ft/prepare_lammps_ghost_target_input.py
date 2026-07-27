from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ELEMENT_ORDER = ("Li", "F", "S", "N", "O", "C", "H")
DEFAULT_MASSES = {
    "Li": 6.94,
    "F": 18.998403163,
    "S": 32.06,
    "N": 14.007,
    "O": 15.999,
    "C": 12.011,
    "H": 1.008,
}


@dataclass(frozen=True)
class PDBAtom:
    index: int
    serial: int
    name: str
    residue: str
    residue_id: int
    element: str
    x: float
    y: float
    z: float


def parse_csv_ints(value: str) -> tuple[int, ...]:
    items = tuple(int(piece.strip()) for piece in value.split(",") if piece.strip())
    if not items:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    return items


def parse_element_order(value: str) -> tuple[str, ...]:
    order = tuple(piece.strip() for piece in value.split(",") if piece.strip())
    if not order:
        raise argparse.ArgumentTypeError("--element-order must not be empty")
    if len(set(order)) != len(order):
        raise argparse.ArgumentTypeError("--element-order must not contain duplicates")
    for element in order:
        if element not in DEFAULT_MASSES:
            raise argparse.ArgumentTypeError(f"Unsupported element in --element-order: {element}")
    return order


def element_from_pdb_atom_name(name: str) -> str:
    letters = "".join(ch for ch in name.strip() if ch.isalpha())
    if not letters:
        raise ValueError(f"Cannot infer element from empty PDB atom name {name!r}")
    upper = letters.upper()
    if upper.startswith("LI"):
        return "Li"
    element = upper[0]
    if element not in DEFAULT_MASSES:
        raise ValueError(f"Unsupported element inferred from PDB atom name {name!r}: {element}")
    return element


def parse_cryst1(line: str) -> tuple[float, float, float, float, float, float]:
    fields = line.split()
    if len(fields) < 7:
        raise ValueError(f"Malformed CRYST1 record: {line.rstrip()}")
    return tuple(float(value) for value in fields[1:7])


def parse_pdb(path: Path) -> tuple[list[PDBAtom], tuple[float, float, float]]:
    atoms: list[PDBAtom] = []
    cell: tuple[float, float, float] | None = None

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "CRYST1":
                a, b, c, alpha, beta, gamma = parse_cryst1(line)
                if any(abs(angle - 90.0) > 1e-6 for angle in (alpha, beta, gamma)):
                    raise ValueError(
                        "Only orthorhombic PDB cells are supported for this LAMMPS "
                        f"writer; got angles {(alpha, beta, gamma)}."
                    )
                cell = (a, b, c)
            elif record in {"ATOM", "HETATM"}:
                name = line[12:16].strip()
                residue = line[17:20].strip()
                residue_id_text = line[22:26].strip()
                atom = PDBAtom(
                    index=len(atoms),
                    serial=int(line[6:11]),
                    name=name,
                    residue=residue,
                    residue_id=int(residue_id_text) if residue_id_text else 0,
                    element=element_from_pdb_atom_name(name),
                    x=float(line[30:38]),
                    y=float(line[38:46]),
                    z=float(line[46:54]),
                )
                atoms.append(atom)

    if cell is None:
        raise ValueError(f"No CRYST1 record found in {path}")
    if not atoms:
        raise ValueError(f"No atoms found in {path}")
    return atoms, cell


def wrap_position(value: float, length: float) -> float:
    wrapped = math.fmod(value, length)
    if wrapped < 0.0:
        wrapped += length
    return wrapped


def write_lammps_data(
    *,
    path: Path,
    atoms: list[PDBAtom],
    cell: tuple[float, float, float],
    element_order: tuple[str, ...],
    target_indices: tuple[int, ...],
    target_type: int,
) -> dict[str, object]:
    base_types = {element: i + 1 for i, element in enumerate(element_order)}
    target_set = set(target_indices)
    n_atom_types = max(len(element_order), target_type)

    if target_type in base_types.values():
        raise ValueError(f"target_type={target_type} overlaps with base element types {base_types}.")

    for atom in atoms:
        if atom.element not in base_types:
            raise ValueError(f"Atom index {atom.index} has element {atom.element}, not in {element_order}.")

    for target_index in target_indices:
        if target_index < 0 or target_index >= len(atoms):
            raise ValueError(f"Target index {target_index} is outside [0, {len(atoms) - 1}]")
        if atoms[target_index].element != "Li":
            raise ValueError(f"Target index {target_index} is {atoms[target_index].element}, expected Li.")

    type_elements: list[str] = []
    for atom_type in range(1, n_atom_types + 1):
        if atom_type == target_type:
            type_elements.append("Li")
        elif 1 <= atom_type <= len(element_order):
            type_elements.append(element_order[atom_type - 1])
        else:
            raise ValueError(f"No element mapping for LAMMPS atom type {atom_type}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("LAMMPS data generated from PDB for MatterSim ghost-target FEP-TI\n\n")
        handle.write(f"{len(atoms)} atoms\n")
        handle.write(f"{n_atom_types} atom types\n\n")
        handle.write(f"0.0 {cell[0]:.10f} xlo xhi\n")
        handle.write(f"0.0 {cell[1]:.10f} ylo yhi\n")
        handle.write(f"0.0 {cell[2]:.10f} zlo zhi\n\n")
        handle.write("Masses\n\n")

        for atom_type, element in enumerate(type_elements, start=1):
            handle.write(f"{atom_type} {DEFAULT_MASSES[element]:.12g} # {element}\n")

        handle.write("\nAtoms # atomic\n\n")

        for atom in atoms:
            atom_type = target_type if atom.index in target_set else base_types[atom.element]
            x = wrap_position(atom.x, cell[0])
            y = wrap_position(atom.y, cell[1])
            z = wrap_position(atom.z, cell[2])
            handle.write(
                f"{atom.index + 1} {atom_type} {x:.10f} {y:.10f} {z:.10f} "
                f"# pdb_index={atom.index} pdb_serial={atom.serial} "
                f"name={atom.name} residue={atom.residue}{atom.residue_id}\n"
            )

    return {
        "natoms": len(atoms),
        "cell": cell,
        "element_order": element_order,
        "target_indices_zero_based": target_indices,
        "target_lammps_ids": tuple(index + 1 for index in target_indices),
        "target_type": target_type,
        "base_types": base_types,
        "pair_coeff_elements": type_elements,
    }


def write_lammps_input(
    *,
    path: Path,
    data_path: Path,
    model_path: Path,
    pair_coeff_elements: list[str],
    temperature: float,
    timestep_fs: float,
    friction_fs_inv: float,
    warmup_steps: int,
    steps: int,
    thermo_interval: int,
    dump_interval: int,
    xtc_dump_interval: int,
    seed: int,
    init_velocities: bool,
    final_data_path: Path,
    dump_path: Path,
    xtc_path: Path,
    temperature_log_path: Path | None = None,
    temperature_log_interval: int = 1,
) -> None:
    if timestep_fs <= 0.0:
        raise ValueError("timestep_fs must be positive")
    if friction_fs_inv <= 0.0:
        raise ValueError("friction_fs_inv must be positive")
    if warmup_steps < 0 or steps < 0:
        raise ValueError("warmup_steps and steps must be non-negative")
    if thermo_interval <= 0 or dump_interval <= 0 or xtc_dump_interval <= 0:
        raise ValueError("thermo_interval, dump_interval, and xtc_dump_interval must be positive")
    if thermo_interval <= 0 or dump_interval <= 0:
        raise ValueError("thermo_interval and dump_interval must be positive")
    if temperature_log_path is not None and temperature_log_interval <= 0:
        raise ValueError("temperature_log_interval must be positive")

    timestep_ps = timestep_fs * 0.001
    damping_ps = 1.0 / friction_fs_inv * 0.001
    pair_coeff = " ".join(pair_coeff_elements)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        handle.write("units           metal\n")
        handle.write("atom_style      atomic\n")
        handle.write("boundary        p p p\n\n")
        handle.write(f"variable        DATA_PATH string {data_path}\n")
        handle.write(f"variable        MODEL_PATH string {model_path}\n\n")
        handle.write("newton          on\n\n")
        handle.write("read_data       ${DATA_PATH}\n\n")
        handle.write("pair_style      mliap unified ${MODEL_PATH}\n")
        handle.write(f"pair_coeff      * * {pair_coeff}\n\n")

        if init_velocities:
            handle.write(f"#velocity        all create {temperature:.12g} {seed} mom yes rot yes dist gaussian\n")

        handle.write(f"timestep        {timestep_ps:.12g}\n")
        handle.write("fix             int all nve\n")
        handle.write(
            f"fix             therm all temp/csvr {temperature:.12g} {temperature:.12g} "
            f"{damping_ps:.12g} {seed + 7919}\n\n"
        )
        handle.write("neighbor        2.0 bin\n")
        handle.write("neigh_modify    every 1 delay 0 check yes\n\n")
        if temperature_log_path is not None:
            handle.write("variable        fep_ti_log_step equal step\n")
            handle.write("variable        fep_ti_log_temp equal temp\n")
            handle.write(
                f"fix             fep_ti_temp_log all print {temperature_log_interval} "
                f"\"${{fep_ti_log_step}},${{fep_ti_log_temp}}\" "
                f"file {temperature_log_path} screen no title \"step,temperature_K\"\n\n"
            )
        handle.write(f"thermo          {thermo_interval}\n")
        handle.write("thermo_style    custom step temp pe ke etotal press\n")
        handle.write("thermo_modify   flush yes\n\n")

        if warmup_steps > 0:
            handle.write(f"run             {warmup_steps}\n")
            handle.write("reset_timestep  0\n\n")

        if steps > 0:
            handle.write(f"dump            traj all custom {dump_interval} {dump_path} id type x y z\n")
            handle.write("dump_modify     traj sort id\n")
            handle.write(f"dump            xtc_traj all xtc {xtc_dump_interval} {xtc_path}\n")
            handle.write("dump_modify     xtc_traj sort id\n")
            handle.write(f"run             {steps}\n")
            handle.write("undump          traj\n")
            handle.write("undump          xtc_traj\n")
        else:
            handle.write("run             0\n")

        handle.write(f"write_data      {final_data_path}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare LAMMPS atomic data and input for MatterSim ghost-target FEP-TI from an electrolyte PDB."
    )
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--target-indices", type=parse_csv_ints, default=(0,))
    parser.add_argument("--target-type", type=int, default=8)
    parser.add_argument("--element-order", type=parse_element_order, default=DEFAULT_ELEMENT_ORDER)
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--timestep-fs", type=float, default=0.5)
    parser.add_argument("--friction-fs-inv", type=float, default=0.02)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--thermo-interval", type=int, default=100)
    parser.add_argument("--dump-interval", type=int, default=2000)
    parser.add_argument("--xtc-dump-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--init-velocities", dest="init_velocities", action="store_true", default=True)
    parser.add_argument("--no-init-velocities", dest="init_velocities", action="store_false")
    parser.add_argument("--final-data", type=Path, required=True)
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--temperature-log", type=Path, default=None)
    parser.add_argument("--temperature-log-interval", type=int, default=1)
    parser.add_argument("--xtc-dump", type=Path, required=True)

    args = parser.parse_args()

    if args.target_type <= 0:
        parser.error("--target-type must be positive")

    if args.temperature_log_interval <= 0:
        parser.error("--temperature-log-interval must be positive")
    return args


def main() -> None:
    args = parse_args()

    atoms, cell = parse_pdb(args.pdb)

    metadata = write_lammps_data(
        path=args.data,
        atoms=atoms,
        cell=cell,
        element_order=args.element_order,
        target_indices=args.target_indices,
        target_type=args.target_type,
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
        seed=args.seed,
        init_velocities=args.init_velocities,
        final_data_path=args.final_data.resolve(),
        dump_path=args.dump.resolve(),
        xtc_path=args.xtc_dump.resolve(),
        temperature_log_path=(
            None if args.temperature_log is None else args.temperature_log.resolve()
        ),
        temperature_log_interval=args.temperature_log_interval,
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
        "seed": args.seed,
        "init_velocities": args.init_velocities,
        "dump": str(args.dump),
        "xtc_dump": str(args.xtc_dump),
        "temperature_log": None if args.temperature_log is None else str(args.temperature_log),
        "temperature_log_interval": args.temperature_log_interval,
        **metadata,
    }

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {args.data}")
    print(f"wrote {args.input}")
    print(f"wrote {args.metadata}")
    print(f"pair_coeff * * {' '.join(metadata['pair_coeff_elements'])}")


if __name__ == "__main__":
    main()
