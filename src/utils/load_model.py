import importlib

def load_model_class(model_name: str):
    module_name = model_name[:-4].upper() + model_name[-4:]
    module_path = f"src.models.architectures.{model_name.lower()}"
    try:
        module = importlib.import_module(module_path)
        model_class = getattr(module, module_name)
        return model_class
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import {module_name} from {module_path}: {e}")