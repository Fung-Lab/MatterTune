from matbench.bench import (  # type: ignore[reportMissingImports] # noqa
    MatbenchBenchmark,
)

mb = MatbenchBenchmark.from_file("./results/orb-matbench_mp_gap-results.json.gz")
scores = mb.matbench_mp_gap.scores

print(scores)
