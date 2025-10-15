# quant_project/data/data_cache.py
# 數據緩存管理器

import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import hashlib

logger = logging.getLogger(__name__)


class DataCache:
    """
    管理歷史數據的本地緩存，減少API調用次數
    """

    def __init__(self, cache_dir: str = "data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """加載緩存元數據"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"加載元數據失敗: {e}")
                return {}
        return {}

    def _save_metadata(self):
        """保存緩存元數據"""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"保存元數據失敗: {e}")

    def _get_cache_key(
        self, symbol: str, resolution: str, start_date: str, end_date: str
    ) -> str:
        """生成緩存鍵"""
        key_string = f"{symbol}_{resolution}_{start_date}_{end_date}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """獲取緩存文件路徑"""
        return self.cache_dir / f"{cache_key}.parquet"

    def get(
        self, symbol: str, resolution: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        從緩存獲取數據

        :param symbol: 股票代碼
        :param resolution: 時間週期
        :param start_date: 開始日期
        :param end_date: 結束日期
        :return: DataFrame或None
        """
        cache_key = self._get_cache_key(symbol, resolution, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        if cache_key in self.metadata and cache_path.exists():
            cache_info = self.metadata[cache_key]

            # 檢查緩存是否過期（默認7天）
            cached_time = datetime.fromisoformat(cache_info["cached_at"])
            if datetime.now() - cached_time < timedelta(days=7):
                try:
                    df = pd.read_parquet(cache_path)
                    logger.info(f"從緩存加載 {symbol} 數據 ({len(df)} 筆記錄)")
                    return df
                except Exception as e:
                    logger.error(f"讀取緩存文件失敗: {e}")
                    return None
            else:
                logger.info(f"緩存已過期: {symbol}")
                self._remove_cache(cache_key)

        return None

    def set(
        self,
        symbol: str,
        resolution: str,
        start_date: str,
        end_date: str,
        data: pd.DataFrame,
    ):
        """
        將數據保存到緩存

        :param symbol: 股票代碼
        :param resolution: 時間週期
        :param start_date: 開始日期
        :param end_date: 結束日期
        :param data: 要緩存的DataFrame
        """
        if data.empty:
            return

        cache_key = self._get_cache_key(symbol, resolution, start_date, end_date)
        cache_path = self._get_cache_path(cache_key)

        try:
            # 保存為parquet格式（高效壓縮）
            data.to_parquet(cache_path)

            # 更新元數據
            self.metadata[cache_key] = {
                "symbol": symbol,
                "resolution": resolution,
                "start_date": start_date,
                "end_date": end_date,
                "cached_at": datetime.now().isoformat(),
                "records": len(data),
                "file_size": cache_path.stat().st_size,
            }
            self._save_metadata()

            logger.info(f"已緩存 {symbol} 數據 ({len(data)} 筆記錄)")
        except Exception as e:
            logger.error(f"保存緩存失敗: {e}")

    def _remove_cache(self, cache_key: str):
        """移除特定緩存"""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            cache_path.unlink()
        if cache_key in self.metadata:
            del self.metadata[cache_key]
            self._save_metadata()

    def clear_expired(self, days: int = 7):
        """清理過期緩存"""
        expired_keys = []
        for cache_key, info in self.metadata.items():
            cached_time = datetime.fromisoformat(info["cached_at"])
            if datetime.now() - cached_time > timedelta(days=days):
                expired_keys.append(cache_key)

        for key in expired_keys:
            self._remove_cache(key)

        logger.info(f"清理了 {len(expired_keys)} 個過期緩存")

    def get_cache_stats(self) -> Dict[str, Any]:
        """獲取緩存統計信息"""
        total_size = 0
        total_records = 0
        set()

        for info in self.metadata.values():
            total_size += info.get("file_size", 0)
            total_records += info.get("records", 0)
            symbols.add(info["symbol"])

        return {
            "total_files": len(self.metadata),
            "total_size_mb": round(total_size / 1024 / 1024, 2),
            "total_records": total_records,
            "unique_symbols": len(symbols),
            "symbols": list(symbols),
        }

    def merge_data(
        self, symbol: str, resolution: str, data_list: list[pd.DataFrame]
    ) -> pd.DataFrame:
        """
        合併多個數據片段，去除重複

        :param symbol: 股票代碼
        :param resolution: 時間週期
        :param data_list: DataFrame列表
        :return: 合併後的DataFrame
        """
        if not data_list:
            return pd.DataFrame()

        # 合併所有數據
        merged_df = pd.concat(data_list, axis=0)

        # 去除重複（保留最新的）
        merged_df = merged_df[~merged_df.index.duplicated(keep="last")]

        # 按時間排序
        merged_df.sort_index(inplace=True)

        logger.info(
            f"合併 {symbol} 數據: {len(data_list)} 個片段 -> {len(merged_df)} 筆記錄"
        )
        return merged_df
