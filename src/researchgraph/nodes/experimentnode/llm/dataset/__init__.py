import importlib

def dynamic_dataloader(dataset_name):
    """
    Dynamically imports a dataset module by the specified name.
    translate dataset_name -> module_name
    
    Args:
        module_name (str): The name of the module to import (e.g., "aaa", "bbb").
    
    Returns:
        module: The imported module object.
    """
    if dataset_name == "openai/gsm8k":
        return importlib.import_module(f"{__name__}.gsm8k")
    else:
        ImportError(f"dataset {dataset_name} not implemented")