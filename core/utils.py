# core/utils.py

import logging
import uuid
from datetime import datetime, timezone
from core import config

def setup_logging(level: str = config.LOG_LEVEL) -> None:
    """
    為應用程式配置基礎日誌。
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # 抑制來自底層套件的過於冗長的日誌
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("alpaca").setLevel(logging.INFO)

def generate_client_order_id() -> str:
    """
    生成一個唯一的客戶端訂單 ID。
    """
    return str(uuid.uuid4())

def get_current_timestamp() -> datetime:
    """
    返回當前的 UTC 時間戳。
    """
    return datetime.now(timezone.utc)
