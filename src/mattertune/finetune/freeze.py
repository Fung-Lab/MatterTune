# from __future__ import annotations

# import nshconfig as C


# class ModelFreezeConfig(C.Config):
#     """
#     Configuration for model parameter freezing during fine-tuning.
#     """
    
#     freeze_backbone: bool | None = None
#     """
#     Whether to freeze the backbone model parameters during fine-tuning.
#     If set to True, this will freeze the entire backbone model and override any other freezing settings.
#     """
    
#     freeze_group_bys: list[str] | None = None
#     """
#     Parameter names to freeze during fine-tuning.
#     For example, if "backbone.conv.1" is included, all parameters under that module will be frozen.
#     """
    
    
    