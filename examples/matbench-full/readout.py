from matbench.bench import (  # type: ignore[reportMissingImports] # noqa
    MatbenchBenchmark,
)

mb = MatbenchBenchmark.from_file("/home/lkong88/MatterTune/examples/matbench-full/results/jmp-matbench_mp_gap-fold0-results.json.gz")
scores = mb.matbench_mp_gap.results["fold_0"]["scores"]

print(scores)
