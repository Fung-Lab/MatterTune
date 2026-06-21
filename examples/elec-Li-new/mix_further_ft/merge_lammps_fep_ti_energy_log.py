from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ENERGY_LOG_FIELDS = (
    "step",
    "time_fs",
    "time_ps",
    "temperature_K",
    "mixed_energy_eV",
    "E_I_eV",
    "E_F_with_LJ_eV",
    "E_F_without_LJ_eV",
    "E_LJ_eV",
    "deltaE_with_LJ_eV",
    "deltaE_without_LJ_eV",
)

TIME_FIELDS = {"time_fs", "time_ps"}
FLOAT_FIELDS = set(ENERGY_LOG_FIELDS) - {"step"}


def format_float(value: str, *, is_time: bool = False) -> str:
    if value == "":
        return ""
    number = float(value)
    if math.isnan(number):
        return ""
    return f"{number:.12g}" if is_time else f"{number:.16g}"


def read_temperature_values(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        if "temperature_K" not in reader.fieldnames:
            raise ValueError(f"{path} does not contain a temperature_K column.")
        return [format_float(row.get("temperature_K", "")) for row in reader]


def merge_logs(*, energy_log_raw: Path, temperature_log: Path, output: Path) -> None:
    with energy_log_raw.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{energy_log_raw} is empty.")
        missing = set(ENERGY_LOG_FIELDS) - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{energy_log_raw} is missing columns: {', '.join(sorted(missing))}"
            )
        rows = list(reader)

    temperature_values = read_temperature_values(temperature_log)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=ENERGY_LOG_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for index, row in enumerate(rows):
            merged: dict[str, str] = {}
            for field in ENERGY_LOG_FIELDS:
                value = row.get(field, "")
                if field == "temperature_K":
                    value = (
                        temperature_values[index]
                        if index < len(temperature_values)
                        else ""
                    )
                elif field in FLOAT_FIELDS:
                    value = format_float(value, is_time=field in TIME_FIELDS)
                merged[field] = value
            writer.writerow(merged)

    print(
        f"wrote {output} rows={len(rows)} "
        f"temperature_rows={len(temperature_values)}"
    )
    if len(temperature_values) < len(rows):
        print(
            "warning: fewer temperature rows than energy rows; missing temperatures "
            "were written as empty fields."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the ML-IAP FEP-TI energy CSV with the per-step LAMMPS "
            "temperature sidecar."
        )
    )
    parser.add_argument("--energy-log-raw", type=Path, required=True)
    parser.add_argument("--temperature-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_logs(
        energy_log_raw=args.energy_log_raw,
        temperature_log=args.temperature_log,
        output=args.output,
    )


if __name__ == "__main__":
    main()
