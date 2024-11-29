# Training Configuration Guide

MatterTune uses a comprehensive configuration system to control all aspects of training. This guide covers the key components and how to use them effectively.

## Model Configuration

Control the model architecture and training parameters:

```python
model = mt.configs.JMPBackboneConfig(
    # Specify pre-trained model checkpoint
    ckpt_path="path/to/pretrained/model.pt",

    # Define properties to predict
    properties=[
        mt.configs.EnergyPropertyConfig(
            loss=mt.configs.MAELossConfig(),
            loss_coefficient=1.0
        ),
        mt.configs.ForcesPropertyConfig(
            loss=mt.configs.MAELossConfig(),
            loss_coefficient=10.0,
            conservative=True  # Use energy-conserving force prediction
        )
    ],

    # Configure optimizer
    optimizer=mt.configs.AdamWConfig(lr=1e-4),

    # Optional: Configure learning rate scheduler
    lr_scheduler=mt.configs.CosineAnnealingLRConfig(
        T_max=100,  # Number of epochs
        eta_min=1e-6  # Minimum learning rate
    )
)
```

## Data Configuration

Configure data loading and processing:

```python
data = mt.configs.AutoSplitDataModuleConfig(
    # Specify dataset source
    dataset=mt.configs.XYZDatasetConfig(
        src="path/to/your/data.xyz"
    ),

    # Control data splitting
    train_split=0.8,  # 80% for training

    # Configure batch size and loading
    batch_size=32,
    num_workers=4,  # Number of data loading workers
    pin_memory=True  # Optimize GPU transfer
)
```

## Training Process Configuration

Control the training loop behavior:

```python
trainer = mt.configs.TrainerConfig(
    # Hardware configuration
    accelerator="gpu",
    devices=[0, 1],  # Use GPUs 0 and 1

    # Training stopping criteria
    max_epochs=100,
    # OR: max_steps=1000,  # Stop after 1000 steps
    # OR: max_time=datetime.timedelta(hours=1),  # Stop after 1 hour

    # Validation frequency
    check_val_every_n_epoch=1,

    # Gradient clipping: Prevent exploding gradients
    gradient_clip_val=1.0,

    # Early stopping configuration
    early_stopping=mt.configs.EarlyStoppingConfig(
        monitor="val/energy_mae",
        patience=20,
        mode="min"
    ),

    # Model checkpointing
    checkpoint=mt.configs.ModelCheckpointConfig(
        monitor="val/energy_mae",
        save_top_k=1,
        mode="min"
    ),

    # Configure logging
    loggers=[
        mt.configs.WandbLoggerConfig(
            project="my-project",
            name="experiment-1"
        )
    ]
)

# Combine all configurations
config = mt.configs.MatterTunerConfig(
    model=model,
    data=data,
    trainer=trainer
)
```

## Configuration Management

MatterTune uses [`nshconfig`](https://github.com/nimashoghi/nshconfig) for configuration management, providing several ways to create and load configurations:

### 1. Direct Construction

```python
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(...),
    data=mt.configs.AutoSplitDataModuleConfig(...),
    trainer=mt.configs.TrainerConfig(...)
)
```

### 2. Loading from Files/Dictionaries

```python
# Load from YAML
config = mt.configs.MatterTunerConfig.from_yaml('/path/to/config.yaml')

# Load from JSON
config = mt.configs.MatterTunerConfig.from_json('/path/to/config.json')

# Load from dictionary
config = mt.configs.MatterTunerConfig.from_dict({
    'model': {...},
    'data': {...},
    'trainer': {...}
})
```

### 3. Using Draft Configs

```python
# Create a draft config
config = mt.configs.MatterTunerConfig.draft()

# Set values progressively
config.model = mt.configs.JMPBackboneConfig.draft()
config.model.ckpt_path = "path/to/model.pt"
# ... set other values ...

# Finalize the config
final_config = config.finalize()
```

For more advanced configuration management features, see the [nshconfig documentation](https://github.com/nimashoghi/nshconfig).
