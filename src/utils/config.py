import yaml

def deep_update(orig, new):
    for key, val in new.items():
        if isinstance(val, dict) and key in orig:
            deep_update(orig[key], val)
        else:
            orig[key] = val

def load_config(files):
    config = {}
    for file in files:
        with open(file) as f:
            new_cfg = yaml.safe_load(f)
            deep_update(config, new_cfg)
    return config