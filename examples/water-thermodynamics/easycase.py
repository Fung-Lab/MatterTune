from __future__ import annotations

import mattertune.configs as MC

def main():
    def hparams():
        hparams = MC.MatterTunerConfig.draft()
        hparams.model = MC.MatterSimBackboneConfig.draft()
        hparams.model.graph_convertor = MC.MatterSimGraphConvertorConfig.draft()
        hparams.model.pretrained_model = "MatterSim-v1.0.0-1M"
        hparams.model.optimizer = MC.AdamConfig(lr=8e-5)

        # Add model properties
        hparams.model.properties = []
        energy = MC.EnergyPropertyConfig(
            loss=MC.MSELossConfig(), loss_coefficient=1.0
        )
        hparams.model.properties.append(energy)
        forces = MC.ForcesPropertyConfig(
            loss=MC.MSELossConfig(), conservative=True, loss_coefficient=1.0
        )
        hparams.model.properties.append(forces)

        ## Data Hyperparameters
        hparams.data = MC.ManualSplitDataModuleConfig.draft()
        hparams.data.train = MC.XYZDatasetConfig.draft()
        hparams.data.train.src = "./data/train_water_1000_eVAng.xyz"
        hparams.data.train.down_sample = 30
        hparams.data.train.down_sample_refill = True
        hparams.data.validation = MC.XYZDatasetConfig.draft()
        hparams.data.validation.src = "./data/val_water_1000_eVAng.xyz"
        hparams.data.batch_size = 16

        # ## Trainer Hyperparameters
        # hparams.trainer = MC.TrainerConfig.draft()
        # hparams.trainer.max_epochs = 100
        # hparams.trainer.accelerator = "gpu"
        # hparams.trainer.devices = [0, 1, 2, 3]
        # hparams.trainer.strategy = "ddp"
        # hparams.trainer.gradient_clip_algorithm = "norm"
        # hparams.trainer.gradient_clip_val = 1.0
        # hparams.trainer.precision = "32"

        # # Additional trainer settings
        # hparams.trainer.additional_trainer_kwargs = {
        #     "inference_mode": False,
        # }

        hparams = hparams.finalize(strict=False)
        return hparams

    mt_config = hparams()
    
    ## NOTE: 
    ## Export the config to a json file before running the training
    ## Actually the trainer setting is not necessary so you can skip lines from 55 to 68
    mt_config.to_json_file(
        "./config.json",
    )
    
    # model, trainer = MatterTuner(mt_config).tune()
    

if __name__ == "__main__":
    main()
