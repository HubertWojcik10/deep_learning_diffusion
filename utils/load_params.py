import json
from typing import Dict

def load_params(params_path: str) -> Dict[str, str]:
    """
        Load parameters from the given path
    """

    with open(params_path, 'r') as f:
        params = json.load(f)

    return params