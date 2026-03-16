import yaml
import logging
from pathlib import Path

logger = logging.getLogger("ConfigLoader")

def load_config(config_path="config.yaml"):
    """
    Tải cấu hình từ file YAML.
    """
    path = Path(config_path)
    if not path.exists():
        logger.error(f"Khong tim thay file cau hinh tai: {config_path}")
        return {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Loi khi tai file cau hinh: {e}")
        return {}
