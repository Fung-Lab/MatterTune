

data_module = ABC-MatterTuneBaseDataModule

backbone = load_backbone()
backbone = GOCStyleBB(backbone)

output_heads = list[OutputHeadConfig]

finetune = FinetuneModuleBase(
    backbone
    output_heads
)

trainer = pl.Trainer(finetune)

<!-- trainer : pl.Lignthning Module : FinetuneModuleBase

FinetuneModuleBase
- __init__
- forward
- train_step
- val_step
- test_step
- predict_step
- save
- load
...
- self.lr_scheduler: basic implementation + protocol, StepLR|Platauo|Cosine
- self.optimizer: basic implementation
- self.monitor: basic implementation

trainer.fit(data_module)

trainer -> Calculator

class Calculator
    def calculate(atoms)
        atoms -> datamodule
        trainer.predict(datamodule)


    -->


## Implementing your own backbone

```python
import mattertune as mt

@mt.backbone_registry.register
class MyBackboneConfig(mt.FinetuneModuleBaseConfig):
    name: Literal["my_backbone"] = "my_backbone"
    # ^ Unique name for the backbone

    # ... your hyperparameters here

    @override
    @classmethod
    def model_cls(cls):
        return MyBackbone


class MyBackbone(mt.FinetuneModuleBase[MyBackboneConfig]):
    # ... your implementation here, must implement
    # the abstract methods from FinetuneModuleBase.

```


## Implementing your own dataset

```python
import mattertune as mt

@mt.dataset_registry.register
class MyDatasetConfig(mt.DatasetConfigBase):
    name: Literal["my_dataset"] = "my_dataset"
    # ^ Unique name for the dataset

    # ... your hyperparameters here

    @override
    @classmethod
    def dataset_cls(cls):
        return MyDataset


class MyDataset(mt.DatasetBase[MyDatasetConfig]):
    # ... your implementation here

```
