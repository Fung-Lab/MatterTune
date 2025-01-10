__codegen__ = True


from mattertune.wrappers.property_predictor import PropertyConfig as PropertyConfig


from . import property_predictor as property_predictor

__all__ = [
    "PropertyConfig",
    "property_predictor",
]
