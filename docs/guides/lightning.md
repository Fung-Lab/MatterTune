# Advanced: Lightning Integration

MatterTune uses PyTorch Lightning as its core training framework. This document outlines how Lightning is integrated and what functionality it provides.

## Core Components

### LightningModule Integration

The base model class `FinetuneModuleBase` inherits from `LightningModule` and provides:

- Automatic device management (GPU/CPU handling)
- Distributed training support
- Built-in training/validation/test loops
- Logging and metrics tracking
- Checkpoint management

```python
class FinetuneModuleBase(LightningModule):
    def training_step(self, batch, batch_idx):
        output = self(batch)
        loss = self._compute_loss(output["predicted_properties"], self.batch_to_labels(batch))
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(batch)
        self._compute_loss(output["predicted_properties"], self.batch_to_labels(batch))

    def test_step(self, batch, batch_idx):
        output = self(batch)
        self._compute_loss(output["predicted_properties"], self.batch_to_labels(batch))

    def configure_optimizers(self):
        return create_optimizer(self.hparams.optimizer, self.parameters())
```

### Data Handling

MatterTune uses Lightning's DataModule system for standardized data loading:

```python
class MatterTuneDataModule(LightningDataModule):
    def prepare_data(self):
        # Download data if needed
        pass

    def setup(self, stage):
        # Create train/val/test splits
        self.datasets = self.hparams.create_datasets()

    def train_dataloader(self):
        return self.lightning_module.create_dataloader(
            self.datasets["train"],
            has_labels=True
        )

    def val_dataloader(self):
        return self.lightning_module.create_dataloader(
            self.datasets["validation"],
            has_labels=True
        )
```

## Key Features

### 1. Checkpoint Management

Lightning automatically handles model checkpointing:

```python
checkpoint_callback = ModelCheckpointConfig(
    monitor="val/forces_mae",
    dirpath="./checkpoints",
    filename="best-model",
    save_top_k=1,
    mode="min"
).create_callback()

trainer = Trainer(callbacks=[checkpoint_callback])
```

### 2. Early Stopping

Built-in early stopping support:

```python
early_stopping = EarlyStoppingConfig(
    monitor="val/forces_mae",
    patience=20,
    mode="min"
).create_callback()

trainer = Trainer(callbacks=[early_stopping])
```

### 3. Multi-GPU Training

Lightning handles distributed training with minimal code changes:

```python
# Single GPU
trainer = Trainer(accelerator="gpu", devices=[0])

# Multiple GPUs with DDP
trainer = Trainer(accelerator="gpu", devices=[0,1], strategy="ddp")
```

### 4. Logging

Lightning provides unified logging interfaces:

```python
def training_step(self, batch, batch_idx):
    loss = ...
    self.log("train_loss", loss)
    self.log_dict({
        "energy_mae": energy_mae,
        "forces_mae": forces_mae
    })
```

### 5. Precision Settings

Easy configuration of precision:

```python
# 32-bit training
trainer = Trainer(precision="32-true")

# Mixed precision training
trainer = Trainer(precision="16-mixed")
```

## Available Trainer Configurations

The `TrainerConfig` class exposes common Lightning Trainer settings:

```python
trainer_config = TrainerConfig(
    # Hardware
    accelerator="gpu",
    devices=[0,1],
    precision="16-mixed",

    # Training
    max_epochs=100,
    gradient_clip_val=1.0,

    # Validation
    val_check_interval=1.0,
    check_val_every_n_epoch=1,

    # Callbacks
    early_stopping=EarlyStoppingConfig(...),
    checkpoint=ModelCheckpointConfig(...),

    # Logging
    loggers=["tensorboard", "wandb"]
)
```

## Best Practices

1. Use `self.log()` for tracking metrics during training
2. Enable checkpointing to save model states
3. Set appropriate early stopping criteria
4. Use appropriate precision settings for your hardware
5. Configure multi-GPU training based on available resources

## Advanced Usage

For advanced use cases:

```python
# Custom training loop
@override
def training_step(self, batch, batch_idx):
    if self.trainer.global_rank == 0:
        # Do something only on main process
        pass

    # Access trainer properties
    if self.trainer.is_last_batch:
        # Special handling for last batch
        pass

# Custom validation
@override
def validation_epoch_end(self, outputs):
    # Compute epoch-level metrics
    pass
```

This integration provides a robust foundation for training atomistic models while handling common ML engineering concerns automatically.
