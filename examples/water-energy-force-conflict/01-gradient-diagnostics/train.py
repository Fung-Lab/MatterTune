from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rich

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))

from common import (  # noqa: E402
    GradientDiagnosticsCallback,
    add_common_args,
    build_config,
    default_device,
    evaluate_checkpoints,
    finalize_common_args,
    make_checkpoint_callbacks,
    prepare_train_data,
    print_run_header,
    run_preflight,
    run_training,
    summarize_gradient_diagnostics,
    summarize_checkpoint_callbacks,
    write_json,
)


EXPERIMENT_NAME = "01-gradient-diagnostics"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Water few-shot fine-tuning with energy/force gradient diagnostics."
    )
    add_common_args(parser, experiment_name=EXPERIMENT_NAME)
    args = parser.parse_args()
    return finalize_common_args(args, experiment_name=EXPERIMENT_NAME)


def main(args: argparse.Namespace) -> None:
    prepared_data = prepare_train_data(
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
        output_dir=Path(args.output_dir),
        train_down_sample=args.train_down_sample,
        down_sample_refill=args.down_sample_refill,
        seed=args.sample_seed,
    )
    print_run_header(args, prepared_data, EXPERIMENT_NAME)
    preflight = run_preflight(args, prepared_data)

    config = build_config(args, prepared_data)
    checkpoint_callbacks = make_checkpoint_callbacks(
        Path(args.checkpoint_dir), labels=["best-total"]
    )
    gradient_path = Path(args.output_dir) / "gradient_diagnostics.jsonl"
    callbacks = [
        GradientDiagnosticsCallback(
            output_path=gradient_path,
            probe_interval=args.grad_probe_interval,
        ),
        *checkpoint_callbacks,
    ]
    _, trainer = run_training(config, callbacks)

    final_ckpt = Path(args.checkpoint_dir) / "final.ckpt"
    trainer.save_checkpoint(final_ckpt)

    checkpoint_paths = summarize_checkpoint_callbacks(checkpoint_callbacks)
    checkpoint_paths["final"] = str(final_ckpt)
    eval_metrics = evaluate_checkpoints(checkpoint_paths=checkpoint_paths, args=args)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "output_dir": str(args.output_dir),
        "device": default_device(args),
        "preflight": preflight,
        "checkpoints": checkpoint_paths,
        "evaluation": eval_metrics,
        "gradient_diagnostics": summarize_gradient_diagnostics(gradient_path),
    }
    metrics_path = Path(args.output_dir) / "metrics.json"
    write_json(metrics_path, metrics)
    rich.print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main(parse_args())
