import yaml

def deep_update(orig, new):
    """
    Recursively update a dictionary with values from another dictionary.
    
    This function performs a deep merge of the `new` dictionary into the `orig` dictionary,
    preserving nested dictionary structures and only overwriting leaf values.
    
    Args:
        orig: Dictionary to be updated in-place
        new: Dictionary containing values to merge into orig
    
    Returns:
        None (modifies orig in-place)
    """
    for key, val in new.items():
        if isinstance(val, dict) and key in orig:
            deep_update(orig[key], val)
        else:
            orig[key] = val

def load_config(files):
    """
    Load and merge YAML configuration files into a single dictionary.
    
    Sequentially loads multiple YAML configuration files and merges them using
    deep_update, allowing for hierarchical configuration overrides. Later files
    override earlier ones.
    
    Args:
        files: List of file paths to YAML configuration files to load
    
    Returns:
        Dictionary containing the merged configuration from all files
    
    Raises:
        FileNotFoundError: If any of the specified files does not exist
        yaml.YAMLError: If any file contains invalid YAML syntax
    """
    config = {}
    for file in files:
        with open(file) as f:
            new_cfg = yaml.safe_load(f)
            deep_update(config, new_cfg)
    return config