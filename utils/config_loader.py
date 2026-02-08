import json
import os
from typing import Dict, Any


def load_secret_config(path: str) -> Dict[str, Any]:
    """Load a JSON config for secrets. Returns empty dict if missing or invalid."""
    if not path:
        return {}
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
