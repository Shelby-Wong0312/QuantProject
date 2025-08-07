# -*- coding: utf-8 -*-
"""
MT4橋接系統配置模組
定義ZeroMQ連接埠設定、MT4符號對應表、交易參數默認值
提供配置管理和驗證功能
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import configparser

logger = logging.getLogger(__name__)

class BrokerType(Enum):
    """經紀商類型枚舉"""
    GENERIC = "GENERIC"
    ALPARI = "ALPARI"
    FXPRO = "FXPRO"
    IC_MARKETS = "IC_MARKETS"
    PEPPERSTONE = "PEPPERSTONE"
    OANDA = "OANDA"

class SymbolType(Enum):
    """交易品種類型枚舉"""
    FOREX = "FOREX"          # 外匯
    METAL = "METAL"          # 貴金屬
    INDEX = "INDEX"          # 指數
    COMMODITY = "COMMODITY"  # 商品
    CRYPTO = "CRYPTO"        # 加密貨幣
    CFD = "CFD"             # 差價合約

@dataclass
class ZeroMQConfig:
    """ZeroMQ配置"""
    # REQ-REP模式端口(命令通訊)
    req_port: int = 5555    # Python -> MT4 請求端口
    rep_port: int = 5556    # MT4 -> Python 回應端口
    
    # PUB-SUB模式端口(數據流)
    pub_port: int = 5557    # MT4 -> Python 發布端口  
    sub_port: int = 5558    # Python <- MT4 訂閱端口
    
    # 連接設置
    bind_address: str = "tcp://*"      # MT4端綁定地址
    connect_address: str = "tcp://localhost"  # Python端連接地址
    
    # 超時設置(毫秒)
    send_timeout: int = 5000
    recv_timeout: int = 5000
    heartbeat_timeout: int = 1000
    
    # 重連設置
    max_retries: int = 3
    retry_delay: float = 1.0  # 秒
    
    def get_req_url(self) -> str:
        """獲取REQ端點URL"""
        return f"{self.connect_address}:{self.req_port}"
    
    def get_sub_url(self) -> str:
        """獲取SUB端點URL"""
        return f"{self.connect_address}:{self.sub_port}"
    
    def get_bind_req_url(self) -> str:
        """獲取REQ綁定URL (MT4端使用)"""
        return f"{self.bind_address}:{self.req_port}"
    
    def get_bind_pub_url(self) -> str:
        """獲取PUB綁定URL (MT4端使用)"""
        return f"{self.bind_address}:{self.pub_port}"

@dataclass
class SymbolConfig:
    """交易品種配置"""
    mt4_symbol: str                    # MT4中的品種名稱
    standard_symbol: str               # 標準化品種名稱  
    symbol_type: SymbolType            # 品種類型
    digits: int = 5                    # 小數位數
    pip_value: float = 1.0            # 點值
    min_lot: float = 0.01             # 最小手數
    max_lot: float = 100.0            # 最大手數
    lot_step: float = 0.01            # 手數步長
    margin_required: float = 1000.0    # 所需保證金
    
    # 交易時間
    trading_hours: Dict[str, str] = field(default_factory=dict)
    
    # 點差設置
    typical_spread: float = 2.0        # 典型點差
    max_spread: float = 10.0          # 最大點差
    
    # 風險管理
    max_risk_per_trade: float = 0.02   # 單筆交易最大風險比例
    
    def __post_init__(self):
        if not self.trading_hours:
            self.trading_hours = {
                "sunday": "22:00-21:00",  # 週日晚10點到週一晚9點
                "monday": "00:00-24:00",
                "tuesday": "00:00-24:00", 
                "wednesday": "00:00-24:00",
                "thursday": "00:00-24:00",
                "friday": "00:00-21:00"   # 週五到晚9點
            }

@dataclass  
class TradingConfig:
    """交易配置"""
    # 風險管理
    max_daily_loss: float = 1000.0        # 每日最大虧損
    max_drawdown_percent: float = 20.0     # 最大回撤百分比
    max_positions: int = 10                # 最大同時持倉數
    max_positions_per_symbol: int = 3      # 每個品種最大持倉數
    
    # 訂單設置
    default_magic_number: int = 12345      # 默認魔術數字
    default_comment: str = "MT4Bridge"     # 默認註釋
    max_slippage: float = 3.0             # 最大滑點(點)
    order_timeout: int = 30                # 訂單超時(秒)
    
    # 資金管理
    risk_per_trade: float = 0.02          # 每筆交易風險比例
    position_sizing_method: str = "FIXED"  # 倉位大小計算方法: FIXED, PERCENT, KELLY
    fixed_lot_size: float = 0.1           # 固定手數
    
    # 交易時間控制
    allow_trading: bool = True
    trading_start_hour: int = 0           # 交易開始時間(小時)
    trading_end_hour: int = 23            # 交易結束時間(小時)
    avoid_news_minutes: int = 30          # 避開重要新聞時間(分鐘)
    
    # 止損止盈設置
    default_stop_loss_pips: float = 50.0  # 默認止損點數
    default_take_profit_pips: float = 100.0  # 默認止盈點數
    use_trailing_stop: bool = False       # 是否使用移動止損
    trailing_stop_pips: float = 20.0      # 移動止損點數

@dataclass
class AccountConfig:
    """賬戶配置"""
    account_number: str = ""              # 賬戶號碼
    account_name: str = ""                # 賬戶名稱
    broker: BrokerType = BrokerType.GENERIC  # 經紀商類型
    account_currency: str = "USD"         # 賬戶貨幣
    leverage: int = 100                   # 槓桿比例
    
    # 監控設置
    monitor_enabled: bool = True          # 是否啟用監控
    monitor_interval: int = 60            # 監控間隔(秒)
    snapshot_interval: int = 300          # 快照間隔(秒)
    
    # 警報設置
    low_balance_threshold: float = 0.1    # 低餘額警報閾值
    margin_level_warning: float = 200.0   # 保證金水平警報
    margin_level_critical: float = 100.0  # 保證金水平嚴重警報
    
    # 報告設置
    daily_report: bool = True             # 是否生成日報
    weekly_report: bool = True            # 是否生成週報
    performance_tracking: bool = True     # 是否追蹤績效

class MT4BridgeConfig:
    """MT4橋接系統主配置類"""
    
    def __init__(self, config_file: str = None):
        """
        初始化配置
        
        Args:
            config_file: 配置文件路徑，默認使用mt4_bridge_config.json
        """
        self.config_file = config_file or self._get_default_config_path()
        
        # 初始化各模組配置
        self.zeromq = ZeroMQConfig()
        self.trading = TradingConfig()
        self.account = AccountConfig()
        self.symbols = {}  # {standard_symbol: SymbolConfig}
        
        # 載入配置
        self.load_config()
        self._setup_default_symbols()
    
    def _get_default_config_path(self) -> str:
        """獲取默認配置文件路徑"""
        # 使用mt4_bridge目錄下的配置文件
        bridge_dir = os.path.dirname(__file__)
        return os.path.join(bridge_dir, "mt4_bridge_config.json")
    
    def load_config(self) -> bool:
        """載入配置文件"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 載入各模組配置
                if 'zeromq' in data:
                    self.zeromq = ZeroMQConfig(**data['zeromq'])
                
                if 'trading' in data:
                    self.trading = TradingConfig(**data['trading'])
                
                if 'account' in data:
                    # 處理枚舉類型
                    account_data = data['account'].copy()
                    if 'broker' in account_data:
                        account_data['broker'] = BrokerType(account_data['broker'])
                    self.account = AccountConfig(**account_data)
                
                if 'symbols' in data:
                    self.symbols = {}
                    for symbol_name, symbol_data in data['symbols'].items():
                        symbol_data = symbol_data.copy()
                        if 'symbol_type' in symbol_data:
                            symbol_data['symbol_type'] = SymbolType(symbol_data['symbol_type'])
                        self.symbols[symbol_name] = SymbolConfig(**symbol_data)
                
                logger.info(f"配置已載入: {self.config_file}")
                return True
            else:
                logger.info("配置文件不存在，使用默認配置")
                self.save_config()  # 創建默認配置文件
                return True
                
        except Exception as e:
            logger.error(f"載入配置失敗: {e}")
            return False
    
    def save_config(self) -> bool:
        """保存配置文件"""
        try:
            # 確保目錄存在
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            # 準備配置數據
            config_data = {
                'zeromq': asdict(self.zeromq),
                'trading': asdict(self.trading),
                'account': asdict(self.account),
                'symbols': {}
            }
            
            # 處理枚舉類型
            config_data['account']['broker'] = self.account.broker.value
            
            # 保存品種配置
            for symbol_name, symbol_config in self.symbols.items():
                symbol_dict = asdict(symbol_config)
                symbol_dict['symbol_type'] = symbol_config.symbol_type.value
                config_data['symbols'][symbol_name] = symbol_dict
            
            # 寫入文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"配置已保存: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置失敗: {e}")
            return False
    
    def _setup_default_symbols(self):
        """設置默認交易品種"""
        if not self.symbols:  # 如果沒有配置品種，添加默認品種
            default_symbols = [
                # 主要外匯對
                SymbolConfig("EURUSD", "EUR/USD", SymbolType.FOREX, digits=5, pip_value=10.0),
                SymbolConfig("GBPUSD", "GBP/USD", SymbolType.FOREX, digits=5, pip_value=10.0),
                SymbolConfig("USDJPY", "USD/JPY", SymbolType.FOREX, digits=3, pip_value=10.0),
                SymbolConfig("USDCHF", "USD/CHF", SymbolType.FOREX, digits=5, pip_value=10.0),
                SymbolConfig("AUDUSD", "AUD/USD", SymbolType.FOREX, digits=5, pip_value=10.0),
                SymbolConfig("USDCAD", "USD/CAD", SymbolType.FOREX, digits=5, pip_value=10.0),
                SymbolConfig("NZDUSD", "NZD/USD", SymbolType.FOREX, digits=5, pip_value=10.0),
                
                # 貴金屬
                SymbolConfig("XAUUSD", "GOLD", SymbolType.METAL, digits=2, pip_value=100.0, 
                           margin_required=5000.0, max_lot=10.0),
                SymbolConfig("XAGUSD", "SILVER", SymbolType.METAL, digits=3, pip_value=50.0,
                           margin_required=3000.0, max_lot=50.0),
                
                # 主要指數
                SymbolConfig("US30", "DOW30", SymbolType.INDEX, digits=1, pip_value=10.0,
                           margin_required=2000.0),
                SymbolConfig("SPX500", "S&P500", SymbolType.INDEX, digits=1, pip_value=10.0,
                           margin_required=2000.0),
                SymbolConfig("NAS100", "NASDAQ", SymbolType.INDEX, digits=1, pip_value=10.0,
                           margin_required=1000.0),
                
                # 石油
                SymbolConfig("USOIL", "WTI", SymbolType.COMMODITY, digits=2, pip_value=100.0,
                           margin_required=1000.0, max_lot=50.0)
            ]
            
            for symbol in default_symbols:
                self.symbols[symbol.standard_symbol] = symbol
    
    def get_symbol_config(self, symbol: str) -> Optional[SymbolConfig]:
        """
        獲取品種配置
        
        Args:
            symbol: 品種名稱(支持MT4名稱或標準名稱)
            
        Returns:
            SymbolConfig: 品種配置，未找到時返回None
        """
        # 首先嘗試標準名稱查找
        if symbol in self.symbols:
            return self.symbols[symbol]
        
        # 然後嘗試MT4名稱查找
        for config in self.symbols.values():
            if config.mt4_symbol == symbol:
                return config
        
        return None
    
    def add_symbol(self, symbol_config: SymbolConfig):
        """添加品種配置"""
        self.symbols[symbol_config.standard_symbol] = symbol_config
        logger.info(f"添加品種配置: {symbol_config.standard_symbol}")
    
    def remove_symbol(self, symbol: str):
        """移除品種配置"""
        if symbol in self.symbols:
            del self.symbols[symbol]
            logger.info(f"移除品種配置: {symbol}")
        else:
            logger.warning(f"品種配置不存在: {symbol}")
    
    def get_mt4_symbol_mapping(self) -> Dict[str, str]:
        """獲取MT4品種名稱映射表"""
        return {config.standard_symbol: config.mt4_symbol for config in self.symbols.values()}
    
    def get_symbols_by_type(self, symbol_type: SymbolType) -> List[SymbolConfig]:
        """根據類型獲取品種列表"""
        return [config for config in self.symbols.values() if config.symbol_type == symbol_type]
    
    def validate_config(self) -> List[str]:
        """
        驗證配置有效性
        
        Returns:
            List[str]: 錯誤信息列表，空列表表示配置有效
        """
        errors = []
        
        # 驗證ZeroMQ端口
        ports = [self.zeromq.req_port, self.zeromq.rep_port, 
                self.zeromq.pub_port, self.zeromq.sub_port]
        if len(set(ports)) != len(ports):
            errors.append("ZeroMQ端口配置重複")
        
        for port in ports:
            if not (1024 <= port <= 65535):
                errors.append(f"端口號無效: {port}")
        
        # 驗證交易配置
        if self.trading.risk_per_trade <= 0 or self.trading.risk_per_trade > 1:
            errors.append(f"風險比例無效: {self.trading.risk_per_trade}")
        
        if self.trading.max_positions <= 0:
            errors.append(f"最大持倉數無效: {self.trading.max_positions}")
        
        # 驗證品種配置
        for symbol_name, config in self.symbols.items():
            if config.min_lot <= 0 or config.min_lot > config.max_lot:
                errors.append(f"品種 {symbol_name} 手數配置無效")
            
            if config.digits < 0 or config.digits > 8:
                errors.append(f"品種 {symbol_name} 小數位數無效: {config.digits}")
        
        return errors
    
    def get_config_summary(self) -> Dict[str, Any]:
        """獲取配置摘要"""
        return {
            "zeromq_ports": {
                "req": self.zeromq.req_port,
                "rep": self.zeromq.rep_port,
                "pub": self.zeromq.pub_port,
                "sub": self.zeromq.sub_port
            },
            "trading": {
                "max_positions": self.trading.max_positions,
                "risk_per_trade": self.trading.risk_per_trade,
                "max_daily_loss": self.trading.max_daily_loss
            },
            "account": {
                "broker": self.account.broker.value,
                "currency": self.account.account_currency,
                "leverage": self.account.leverage
            },
            "symbols_count": len(self.symbols),
            "symbols_by_type": {
                symbol_type.value: len(self.get_symbols_by_type(symbol_type))
                for symbol_type in SymbolType
            }
        }
    
    def reset_to_defaults(self):
        """重置為默認配置"""
        self.zeromq = ZeroMQConfig()
        self.trading = TradingConfig()
        self.account = AccountConfig()
        self.symbols = {}
        self._setup_default_symbols()
        logger.info("配置已重置為默認值")
    
    def export_config(self, export_path: str) -> bool:
        """導出配置到指定路径"""
        try:
            # 創建配置副本
            original_path = self.config_file
            self.config_file = export_path
            
            # 保存到新路徑
            result = self.save_config()
            
            # 恢復原路徑
            self.config_file = original_path
            
            if result:
                logger.info(f"配置已導出到: {export_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"導出配置失敗: {e}")
            return False
    
    def import_config(self, import_path: str) -> bool:
        """從指定路徑導入配置"""
        try:
            if not os.path.exists(import_path):
                logger.error(f"配置文件不存在: {import_path}")
                return False
            
            # 備份當前配置
            backup_config = {
                'zeromq': self.zeromq,
                'trading': self.trading,
                'account': self.account,
                'symbols': self.symbols.copy()
            }
            
            # 嘗試載入新配置
            original_path = self.config_file
            self.config_file = import_path
            
            if self.load_config():
                # 驗證新配置
                errors = self.validate_config()
                if errors:
                    # 配置無效，恢復備份
                    logger.error(f"導入的配置無效: {', '.join(errors)}")
                    self.zeromq = backup_config['zeromq']
                    self.trading = backup_config['trading']
                    self.account = backup_config['account']
                    self.symbols = backup_config['symbols']
                    self.config_file = original_path
                    return False
                
                # 配置有效，保存到原位置
                self.config_file = original_path
                self.save_config()
                logger.info(f"配置已從 {import_path} 導入")
                return True
            else:
                # 載入失敗，恢復備份
                self.zeromq = backup_config['zeromq']
                self.trading = backup_config['trading']
                self.account = backup_config['account']
                self.symbols = backup_config['symbols']
                self.config_file = original_path
                return False
                
        except Exception as e:
            logger.error(f"導入配置失敗: {e}")
            return False


# 全局配置實例
_global_config = None

def get_config() -> MT4BridgeConfig:
    """獲取全局配置實例"""
    global _global_config
    if _global_config is None:
        _global_config = MT4BridgeConfig()
    return _global_config

def reload_config() -> bool:
    """重新載入配置"""
    global _global_config
    if _global_config is not None:
        return _global_config.load_config()
    return False

def save_config() -> bool:
    """保存當前配置"""
    global _global_config
    if _global_config is not None:
        return _global_config.save_config()
    return False

# 便利函數
def get_zeromq_config() -> ZeroMQConfig:
    """獲取ZeroMQ配置"""
    return get_config().zeromq

def get_trading_config() -> TradingConfig:
    """獲取交易配置"""
    return get_config().trading

def get_account_config() -> AccountConfig:
    """獲取賬戶配置"""
    return get_config().account

def get_symbol_config(symbol: str) -> Optional[SymbolConfig]:
    """獲取品種配置"""
    return get_config().get_symbol_config(symbol)

def create_custom_config(config_file: str) -> MT4BridgeConfig:
    """創建自定義配置實例"""
    return MT4BridgeConfig(config_file)