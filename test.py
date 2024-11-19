import mattertune as mt

# Define configuration
config = mt.configs.MatterTunerConfig(
    model=mt.configs.JMPBackboneConfig(
        ckpt_path="path/to/pretrained/model.pt",
        properties=[
            mt.configs.EnergyPropertyConfig(
                loss=mt.configs.MAELossConfig(),
                loss_coefficient=1.0
            )
        ],
        optimizer=mt.configs.AdamWConfig(lr=1e-4)
    ),
    data=mt.configs.AutoSplitDataModuleConfig(
        dataset=mt.configs.XYZDatasetConfig(
            src="path/to/your/data.xyz"
        ),
        train_split=0.8,
        batch_size=32
    )
    trainer=mt.configs.TrainerConfig(
        max_epochs=100,
        loggers="default"
    )
)

# Create tuner and train
tuner = mt.MatterTuner(config)
model, trainer = tuner.tune()

# Save the fine-tuned model
trainer.save_checkpoint("finetuned_model.ckpt")