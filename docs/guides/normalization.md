# Normalization

MatterTune provides flexible property normalization capabilities through the `mattertune.normalization` module. Normalization is crucial for improving training stability and convergence when fine-tuning models, especially for properties that can vary widely in scale.

## Overview

The normalization system consists of:
- A `NormalizationContext` that provides per-batch information needed for normalization
- Multiple normalizer types that can be composed together
- CLI tools for computing normalization parameters from datasets

## Supported Normalizers

### Mean-Standard Deviation Normalization

Normalizes values using mean and standard deviation: `(x - mean) / std`

```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(
        # ... other configs ...
        normalizers={
            "energy": [
                mt.configs.MeanStdNormalizerConfig(
                    mean=-13.6,  # mean of your property
                    std=2.4      # standard deviation
                )
            ]
        }
    ),
    # ... other configs ...
)
```

### RMS Normalization

Normalizes values by dividing by the root mean square value: `x / rms`

```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(
        # ... other configs ...
        normalizers={
            "forces": [
                mt.configs.RMSNormalizerConfig(
                    rms=2.5  # RMS value of your property
                )
            ]
        }
    ),
    # ... other configs ...
)
```

### Per-Atom Reference Normalization

Subtracts composition-weighted atomic reference values. This is particularly useful for energy predictions where you want to remove the baseline atomic contributions.

```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(
        # ... other configs ...
        normalizers={
            "energy": [
                mt.configs.PerAtomReferencingNormalizerConfig(
                    # Option 1: Direct dictionary mapping
                    per_atom_references={
                        1: -13.6,  # H
                        8: -2000.0  # O
                    }
                    # Option 2: List indexed by atomic number
                    # per_atom_references=[0.0, -13.6, 0.0, ..., -2000.0]
                    # Option 3: Path to JSON file
                    # per_atom_references="path/to/references.json"
                )
            ]
        }
    ),
    # ... other configs ...
)
```

## Computing Normalization Parameters

### Per-Atom References

MatterTune provides a CLI tool to compute per-atom reference values using either linear regression or ridge regression:

```bash
python -m mattertune.normalization \
    config.json \
    energy \
    references.json \
    --reference-model linear
```

Arguments:
- `config.json`: Path to your MatterTune configuration file
- `energy`: Name of the property to compute references for
- `references.json`: Output path for the computed references
- `--reference-model`: Model type (`linear` or `ridge`)
- `--reference-model-kwargs`: Optional JSON string of kwargs for the regression model

The tool will:
1. Load your dataset from the config
2. Fit a linear model to predict property values from atomic compositions
3. Save the computed per-atom references to the specified JSON file

## Composing Multiple Normalizers

You can combine multiple normalizers for a single property. They will be applied in sequence:

```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(
        # ... other configs ...
        normalizers={
            "energy": [
                # First subtract atomic references
                mt.configs.PerAtomReferencingNormalizerConfig(
                    per_atom_references="references.json"
                ),
                # Then apply mean-std normalization
                mt.configs.MeanStdNormalizerConfig(
                    mean=0.0,
                    std=1.0
                )
            ]
        }
    ),
    # ... other configs ...
)
```

## Technical Details

All normalizers implement the `NormalizerModule` protocol which requires:
- `normalize(value: torch.Tensor, ctx: NormalizationContext) -> torch.Tensor`
- `denormalize(value: torch.Tensor, ctx: NormalizationContext) -> torch.Tensor`

The `NormalizationContext` provides composition information needed for per-atom normalization:
```python
@dataclass(frozen=True)
class NormalizationContext:
    compositions: torch.Tensor  # shape: (batch_size, num_elements)
```

Each row in `compositions` represents the element counts for one structure, where the index corresponds to the atomic number (e.g., index 1 for hydrogen).

## Implementation Notes

- Normalization is applied automatically during training
- Loss is computed on normalized values for numerical stability
- Predictions are automatically denormalized before metric computation and output
- The property predictor and ASE calculator interfaces return denormalized values

{py:mod}`mattertune.normalization`
- {py:class}`mattertune.normalization.NormalizerModule`
- {py:class}`mattertune.normalization.MeanStdNormalizerConfig`
- {py:class}`mattertune.normalization.RMSNormalizerConfig`
- {py:class}`mattertune.normalization.PerAtomReferencingNormalizerConfig`
