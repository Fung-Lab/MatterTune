# Fine-Tuning a Pre-trained Model

This guide will walk you through fine-tuning a pre-trained model for predicting molecular properties. We'll use a complete example with detailed explanations.

```python
import mattertune as mt
from pathlib import Path

# Step 1: Define the configuration for our fine-tuning process
config = mt.configs.MatterTunerConfig(
    # Configure the model and its training parameters
    model=mt.configs.JMPBackboneConfig(
        # Path to the pre-trained model checkpoint you want to fine-tune
        ckpt_path=Path("YOUR_CHECKPOINT_PATH"),

        # Configure how atomic structures are processed
        graph_computer=mt.configs.JMPGraphComputerConfig(
            # Set pbc=True for periodic systems (crystals), False for molecules
            pbc=True
        ),

        # Define which properties to predict and how to train them
        properties=[
            mt.configs.EnergyPropertyConfig(
                # Use Mean Absolute Error (MAE) loss for energy prediction
                loss=mt.configs.MAELossConfig(),
                # Weight of this loss in the total loss function
                loss_coefficient=1.0
            )
        ],

        # Configure the optimizer for training
        optimizer=mt.configs.AdamWConfig(
            # Learning rate - adjust based on your dataset size and needs
            lr=1e-4
        ),
    ),

    # Configure how data is loaded and processed
    data=mt.configs.AutoSplitDataModuleConfig(
        # Specify the source dataset (XYZ file format in this case)
        dataset=mt.configs.XYZDatasetConfig(
            src=Path("YOUR_XYZFILE_PATH")
        ),
        # Fraction of data to use for training (0.8 = 80% training, 20% validation)
        train_split=0.8,
        # Number of structures to process at once
        batch_size=4,
    ),

    # Configure the training process
    trainer=mt.configs.TrainerConfig(
        # Maximum number of training epochs
        max_epochs=10,
        # Use GPU for training
        accelerator="gpu",
        # Specify which GPU(s) to use (0 = first GPU)
        devices=[0],
    ),
)

# Step 2: Initialize the MatterTuner with our configuration
tuner = mt.MatterTuner(config)

# Step 3: Start the fine-tuning process
# This returns both the trained model and the trainer object
model, trainer = tuner.tune()

# Step 4: Save the fine-tuned model for later use
trainer.save_checkpoint("finetuned_model.ckpt")

# Step 5: Make predictions with the fine-tuned model
# Create a property predictor interface
property_predictor = model.property_predictor()

# Example: predict energy for a structure
from ase import Atoms

# Create a water molecule as an example
water = Atoms('H2O',
              positions=[[0, 0, 0],    # O atom
                        [0, 0, 0.96],  # H atom
                        [0.93, 0, 0]], # H atom
              cell=[10, 10, 10],
              pbc=True)

# Make predictions
predictions = property_predictor.predict([water], ["energy"])
print(f"Predicted energy: {predictions[0]['energy']} eV")
```

## Key Components Explained

1. **Configuration Structure**:
   - `MatterTunerConfig`: The main configuration container
   - `JMPBackboneConfig`: Specifies the model architecture and training parameters
   - `AutoSplitDataModuleConfig`: Handles data loading and splitting
   - `TrainerConfig`: Controls the training process

2. **Property Prediction**:
   - Define what properties to predict using PropertyConfig objects
   - Each property can have its own loss function and weight
   - Common properties: energy, forces, stress tensors

3. **Data Handling**:
   - Supports various input formats (XYZ, ASE databases, etc.)
   - Automatic train/validation splitting
   - Configurable batch size for memory management

## Common Customizations

```python
# Add force prediction
properties=[
    mt.configs.EnergyPropertyConfig(
        loss=mt.configs.MAELossConfig(),
        loss_coefficient=1.0
    ),
    mt.configs.ForcesPropertyConfig(
        loss=mt.configs.MAELossConfig(),
        loss_coefficient=0.1,  # Usually smaller than energy coefficient
        conservative=True  # Ensures forces are energy-conserving
    )
]

# Use multiple GPUs
trainer=mt.configs.TrainerConfig(
    max_epochs=10,
    accelerator="gpu",
    devices=[0, 1],  # Use GPUs 0 and 1
    strategy="ddp"  # Distributed data parallel training
)

# Add logging with Weights & Biases
trainer=mt.configs.TrainerConfig(
    # ... other settings ...
    loggers=[
        mt.configs.WandbLoggerConfig(
            project="my-project",
            name="experiment-1"
        )
    ]
)
```

## Tips for Successful Fine-Tuning

1. **Data Quality**:
   - Ensure your training data is clean and properly formatted
   - Use a reasonable train/validation split (80/20 is common)
   - Consider normalizing your target properties

2. **Training Parameters**:
   - Start with a small learning rate (1e-4 to 1e-5)
   - Monitor validation loss for signs of overfitting
   - Use early stopping to prevent overfitting
   - Adjust batch size based on your GPU memory

3. **Model Selection**:
   - Choose a pre-trained model suitable for your task
   - Consider the model's original training data and your use case
   - Test different backbones if possible

4. **Monitoring Training**:
   - Use logging to track training progress
   - Monitor both training and validation metrics
   - Save checkpoints regularly
