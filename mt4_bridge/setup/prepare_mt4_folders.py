#!/usr/bin/env python3
"""
MT4 資料夾準備腳本 (prepare_mt4_folders.py)

此腳本用於自動創建 MT4 橋接通訊所需的資料夾結構，
設置 Expert Advisors (EA) 和指標的目錄，
並準備檔案橋接和 ZeroMQ 通訊所需的目錄。

使用方法:
    python prepare_mt4_folders.py [--mt4-path PATH] [--force] [--verbose]

選項:
    --mt4-path PATH     指定 MT4 安裝路徑
    --force            強制重新建立已存在的目錄
    --verbose          顯示詳細的操作資訊
"""

import os
import sys
import argparse
import shutil
import platform
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time


class MT4FolderPreparer:
    """MT4 資料夾準備器"""
    
    def __init__(self, mt4_path: str = None, verbose: bool = False, force: bool = False):
        self.mt4_path = mt4_path
        self.verbose = verbose
        self.force = force
        self.system = platform.system()
        
        # 記錄操作結果
        self.created_folders = []
        self.created_files = []
        self.errors = []
        self.warnings = []
        
        # MT4 基本目錄結構
        self.basic_directories = [
            'MQL4',
            'MQL4/Experts',
            'MQL4/Indicators',
            'MQL4/Scripts',
            'MQL4/Include',
            'MQL4/Libraries',
            'MQL4/Files',
            'Profiles',
            'Templates',
            'Tester'
        ]
        
        # 橋接通訊專用目錄
        self.bridge_directories = [
            'MQL4/Files/bridge',
            'MQL4/Files/bridge/commands',
            'MQL4/Files/bridge/responses',
            'MQL4/Files/bridge/data',
            'MQL4/Files/bridge/logs',
            'MQL4/Files/bridge/config',
            'MQL4/Files/bridge/temp'
        ]
        
        # ZeroMQ 相關目錄
        self.zeromq_directories = [
            'MQL4/Libraries/zmq',
            'MQL4/Include/zmq'
        ]
        
        # Python 橋接腳本目錄
        self.python_bridge_directories = [
            'MQL4/Files/bridge/python',
            'MQL4/Files/bridge/python/src',
            'MQL4/Files/bridge/python/config',
            'MQL4/Files/bridge/python/logs'
        ]
    
    def log(self, message: str, level: str = 'info'):
        """記錄訊息"""
        timestamp = time.strftime('%H:%M:%S')
        
        if level == 'error':
            self.errors.append(message)
            print(f"[{timestamp}] ❌ 錯誤: {message}")
        elif level == 'warning':
            self.warnings.append(message)
            print(f"[{timestamp}] ⚠️  警告: {message}")
        elif level == 'success':
            print(f"[{timestamp}] ✅ {message}")
        elif level == 'info':
            if self.verbose:
                print(f"[{timestamp}] ℹ️  {message}")
    
    def find_mt4_installation(self) -> Optional[str]:
        """自動尋找 MT4 安裝路徑"""
        print("🔍 搜尋 MT4 安裝路徑...")
        
        common_paths = []
        
        if self.system == 'Windows':
            common_paths = [
                r"C:\Program Files\MetaTrader 4",
                r"C:\Program Files (x86)\MetaTrader 4",
                r"C:\Program Files\MetaTrader 4 - Capital.com",
                r"C:\Program Files (x86)\MetaTrader 4 - Capital.com",
                os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal"),
            ]
        elif self.system == 'Darwin':  # macOS
            common_paths = [
                "/Applications/MetaTrader 4.app",
                os.path.expanduser("~/Applications/MetaTrader 4.app"),
                os.path.expanduser("~/Library/Application Support/MetaQuotes/Terminal"),
            ]
        
        for path in common_paths:
            if os.path.exists(path):
                # 驗證是否為有效的 MT4 目錄
                if self._is_valid_mt4_path(path):
                    self.log(f"找到 MT4 安裝: {path}", 'success')
                    return path
                else:
                    self.log(f"路徑存在但不是有效的 MT4 安裝: {path}", 'warning')
        
        return None
    
    def _is_valid_mt4_path(self, path: str) -> bool:
        """驗證是否為有效的 MT4 安裝路徑"""
        if self.system == 'Windows':
            return os.path.exists(os.path.join(path, 'terminal.exe'))
        elif self.system == 'Darwin':
            return os.path.exists(os.path.join(path, 'Contents', 'MacOS', 'MetaTrader 4'))
        return False
    
    def create_directory(self, dir_path: str, description: str = "") -> bool:
        """建立目錄"""
        try:
            if os.path.exists(dir_path):
                if self.force:
                    self.log(f"目錄已存在，強制模式下重新建立: {dir_path}", 'info')
                else:
                    self.log(f"目錄已存在: {dir_path}", 'info')
                    return True
            else:
                os.makedirs(dir_path, exist_ok=True)
                self.created_folders.append(dir_path)
                
                desc_text = f" ({description})" if description else ""
                self.log(f"已建立目錄: {os.path.basename(dir_path)}{desc_text}", 'success')
            
            return True
            
        except PermissionError:
            self.log(f"權限不足，無法建立目錄: {dir_path}", 'error')
            return False
        except Exception as e:
            self.log(f"建立目錄 {dir_path} 時發生錯誤: {str(e)}", 'error')
            return False
    
    def create_basic_directories(self) -> bool:
        """建立 MT4 基本目錄結構"""
        print("\n📁 建立基本目錄結構...")
        
        success_count = 0
        
        directory_descriptions = {
            'MQL4': 'MQL4 主目錄',
            'MQL4/Experts': 'Expert Advisors (自動交易程式)',
            'MQL4/Indicators': '自訂技術指標',
            'MQL4/Scripts': '交易腳本',
            'MQL4/Include': '標頭檔案',
            'MQL4/Libraries': '函式庫',
            'MQL4/Files': '檔案交換目錄',
            'Profiles': '圖表設定檔',
            'Templates': '圖表模板',
            'Tester': '策略測試器'
        }
        
        for directory in self.basic_directories:
            full_path = os.path.join(self.mt4_path, directory)
            description = directory_descriptions.get(directory, "")
            
            if self.create_directory(full_path, description):
                success_count += 1
        
        self.log(f"基本目錄建立完成: {success_count}/{len(self.basic_directories)}", 'success')
        return success_count == len(self.basic_directories)
    
    def create_bridge_directories(self) -> bool:
        """建立橋接通訊專用目錄"""
        print("\n🌉 建立橋接通訊目錄...")
        
        success_count = 0
        
        directory_descriptions = {
            'MQL4/Files/bridge': '橋接通訊主目錄',
            'MQL4/Files/bridge/commands': '命令檔案',
            'MQL4/Files/bridge/responses': '回應檔案',
            'MQL4/Files/bridge/data': '市場資料',
            'MQL4/Files/bridge/logs': '日誌檔案',
            'MQL4/Files/bridge/config': '設定檔案',
            'MQL4/Files/bridge/temp': '暫存檔案'
        }
        
        for directory in self.bridge_directories:
            full_path = os.path.join(self.mt4_path, directory)
            description = directory_descriptions.get(directory, "")
            
            if self.create_directory(full_path, description):
                success_count += 1
        
        self.log(f"橋接目錄建立完成: {success_count}/{len(self.bridge_directories)}", 'success')
        return success_count == len(self.bridge_directories)
    
    def create_zeromq_directories(self) -> bool:
        """建立 ZeroMQ 相關目錄"""
        print("\n📡 建立 ZeroMQ 目錄...")
        
        success_count = 0
        
        directory_descriptions = {
            'MQL4/Libraries/zmq': 'ZeroMQ 函式庫',
            'MQL4/Include/zmq': 'ZeroMQ 標頭檔'
        }
        
        for directory in self.zeromq_directories:
            full_path = os.path.join(self.mt4_path, directory)
            description = directory_descriptions.get(directory, "")
            
            if self.create_directory(full_path, description):
                success_count += 1
        
        self.log(f"ZeroMQ 目錄建立完成: {success_count}/{len(self.zeromq_directories)}", 'success')
        return success_count == len(self.zeromq_directories)
    
    def create_python_bridge_directories(self) -> bool:
        """建立 Python 橋接腳本目錄"""
        print("\n🐍 建立 Python 橋接目錄...")
        
        success_count = 0
        
        directory_descriptions = {
            'MQL4/Files/bridge/python': 'Python 橋接腳本主目錄',
            'MQL4/Files/bridge/python/src': 'Python 源程式碼',
            'MQL4/Files/bridge/python/config': 'Python 設定檔',
            'MQL4/Files/bridge/python/logs': 'Python 日誌檔案'
        }
        
        for directory in self.python_bridge_directories:
            full_path = os.path.join(self.mt4_path, directory)
            description = directory_descriptions.get(directory, "")
            
            if self.create_directory(full_path, description):
                success_count += 1
        
        self.log(f"Python 橋接目錄建立完成: {success_count}/{len(self.python_bridge_directories)}", 'success')
        return success_count == len(self.python_bridge_directories)
    
    def create_configuration_files(self) -> bool:
        """建立預設設定檔案"""
        print("\n⚙️ 建立設定檔案...")
        
        configs = {
            'bridge_config.json': self._get_bridge_config(),
            'zeromq_config.json': self._get_zeromq_config(),
            'python_bridge_config.json': self._get_python_bridge_config()
        }
        
        config_dir = os.path.join(self.mt4_path, 'MQL4', 'Files', 'bridge', 'config')
        success_count = 0
        
        for filename, config_data in configs.items():
            filepath = os.path.join(config_dir, filename)
            
            try:
                if os.path.exists(filepath) and not self.force:
                    self.log(f"設定檔已存在: {filename}", 'info')
                    success_count += 1
                    continue
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                
                self.created_files.append(filepath)
                self.log(f"已建立設定檔: {filename}", 'success')
                success_count += 1
                
            except Exception as e:
                self.log(f"建立設定檔 {filename} 時發生錯誤: {str(e)}", 'error')
        
        return success_count == len(configs)
    
    def _get_bridge_config(self) -> Dict:
        """取得橋接通訊設定"""
        return {
            "bridge_type": "file_bridge",
            "polling_interval_ms": 100,
            "command_file": "command.txt",
            "response_file": "response.txt",
            "data_file": "market_data.csv",
            "log_level": "INFO",
            "max_log_files": 10,
            "log_file_size_mb": 10,
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0.0"
        }
    
    def _get_zeromq_config(self) -> Dict:
        """取得 ZeroMQ 設定"""
        return {
            "enabled": False,
            "socket_type": "REQ",
            "address": "tcp://127.0.0.1",
            "port": 5555,
            "timeout_ms": 5000,
            "high_water_mark": 1000,
            "linger_ms": 1000,
            "heartbeat_interval_ms": 30000,
            "heartbeat_timeout_ms": 90000,
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0.0"
        }
    
    def _get_python_bridge_config(self) -> Dict:
        """取得 Python 橋接設定"""
        return {
            "python_executable": sys.executable,
            "bridge_script": "python_side.py",
            "working_directory": os.path.join(self.mt4_path, 'MQL4', 'Files', 'bridge', 'python', 'src'),
            "auto_start": True,
            "restart_on_error": True,
            "max_restart_attempts": 5,
            "log_level": "INFO",
            "dependencies": [
                "pandas",
                "numpy",
                "pyzmq"
            ],
            "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0.0"
        }
    
    def create_readme_files(self) -> bool:
        """建立 README 說明檔案"""
        print("\n📖 建立說明檔案...")
        
        readme_files = {
            'MQL4/Files/bridge/README.md': self._get_bridge_readme(),
            'MQL4/Files/bridge/python/README.md': self._get_python_bridge_readme(),
            'MQL4/Libraries/zmq/README.md': self._get_zeromq_readme()
        }
        
        success_count = 0
        
        for relative_path, content in readme_files.items():
            filepath = os.path.join(self.mt4_path, relative_path)
            
            try:
                if os.path.exists(filepath) and not self.force:
                    self.log(f"說明檔已存在: {os.path.basename(filepath)}", 'info')
                    success_count += 1
                    continue
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.created_files.append(filepath)
                self.log(f"已建立說明檔: {os.path.basename(filepath)}", 'success')
                success_count += 1
                
            except Exception as e:
                self.log(f"建立說明檔時發生錯誤: {str(e)}", 'error')
        
        return success_count == len(readme_files)
    
    def _get_bridge_readme(self) -> str:
        """取得橋接通訊 README 內容"""
        return """# MT4 橋接通訊目錄

此目錄包含 MT4 與外部系統（如 Python）之間的橋接通訊檔案。

## 目錄結構

- `commands/` - 存放來自外部系統的命令檔案
- `responses/` - 存放 MT4 的回應檔案
- `data/` - 存放市場資料和交易資料
- `logs/` - 存放橋接通訊的日誌檔案
- `config/` - 存放設定檔案
- `temp/` - 存放暫存檔案
- `python/` - 存放 Python 橋接腳本

## 使用說明

1. 外部系統將命令寫入 `commands/` 目錄中的檔案
2. MT4 Expert Advisor 定期檢查命令檔案並執行
3. 執行結果寫入 `responses/` 目錄
4. 市場資料和交易資料存放在 `data/` 目錄

## 注意事項

- 請確保 MT4 有足夠權限讀寫這些目錄
- 定期清理暫存檔案以避免磁碟空間不足
- 檢查日誌檔案以診斷通訊問題

---
建立時間: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """
"""
    
    def _get_python_bridge_readme(self) -> str:
        """取得 Python 橋接 README 內容"""
        return """# Python 橋接腳本

此目錄包含 Python 端的橋接通訊腳本。

## 目錄結構

- `src/` - Python 源程式碼
- `config/` - Python 設定檔案
- `logs/` - Python 端日誌檔案

## 環境需求

```bash
pip install pandas numpy pyzmq
```

## 使用方法

1. 配置 `config/` 目錄中的設定檔案
2. 執行主要橋接腳本
3. 監控日誌檔案以確保正常運作

## 設定檔案

- `bridge_config.json` - 橋接通訊基本設定
- `zeromq_config.json` - ZeroMQ 通訊設定
- `python_bridge_config.json` - Python 端特定設定

---
建立時間: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """
"""
    
    def _get_zeromq_readme(self) -> str:
        """取得 ZeroMQ README 內容"""
        return """# ZeroMQ 函式庫目錄

此目錄用於存放 ZeroMQ 相關的 MQL4 函式庫和標頭檔。

## 使用 ZeroMQ

ZeroMQ 提供高效能的訊息傳遞功能，適合即時交易應用。

### 安裝需求

1. 下載 MQL4 適用的 ZeroMQ 函式庫
2. 將 .dll 檔案放入 `Libraries/` 目錄
3. 將 .mqh 標頭檔放入 `Include/zmq/` 目錄

### 基本用法

```mql4
#include <zmq/zmq.mqh>

// 建立 ZeroMQ 連線
// 發送和接收訊息
```

---
建立時間: """ + time.strftime('%Y-%m-%d %H:%M:%S') + """
"""
    
    def set_directory_permissions(self) -> bool:
        """設置目錄權限"""
        print("\n🔒 設置目錄權限...")
        
        # 在 Windows 上，通常不需要特別設置權限
        # MT4 會自動處理 MQL4/Files 目錄的權限
        
        if self.system == 'Windows':
            self.log("Windows 系統：權限由 MT4 自動管理", 'info')
            return True
        
        # macOS 可能需要設置權限
        elif self.system == 'Darwin':
            try:
                import stat
                
                # 設置關鍵目錄為可讀寫
                key_directories = [
                    os.path.join(self.mt4_path, 'MQL4', 'Files'),
                    os.path.join(self.mt4_path, 'MQL4', 'Files', 'bridge')
                ]
                
                for directory in key_directories:
                    if os.path.exists(directory):
                        os.chmod(directory, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
                        self.log(f"已設置權限: {directory}", 'success')
                
                return True
                
            except Exception as e:
                self.log(f"設置權限時發生錯誤: {str(e)}", 'warning')
                return False
        
        return True
    
    def generate_summary_report(self) -> Dict:
        """生成摘要報告"""
        return {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'mt4_path': self.mt4_path,
            'system': self.system,
            'created_folders': self.created_folders,
            'created_files': self.created_files,
            'errors': self.errors,
            'warnings': self.warnings,
            'summary': {
                'total_folders_created': len(self.created_folders),
                'total_files_created': len(self.created_files),
                'errors_count': len(self.errors),
                'warnings_count': len(self.warnings),
                'overall_status': 'SUCCESS' if len(self.errors) == 0 else 'PARTIAL' if len(self.created_folders) > 0 else 'FAILED'
            }
        }
    
    def save_report(self, report: Dict, filepath: str = None):
        """儲存報告"""
        if filepath is None:
            filepath = os.path.join(os.path.dirname(__file__), 'mt4_folder_preparation_report.json')
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.log(f"報告已儲存至: {filepath}", 'success')
            
        except Exception as e:
            self.log(f"儲存報告時發生錯誤: {str(e)}", 'error')
    
    def run_full_preparation(self) -> bool:
        """執行完整的資料夾準備"""
        print("🚀 開始 MT4 資料夾準備...")
        print("=" * 60)
        
        # 1. 確定 MT4 路徑
        if not self.mt4_path:
            self.mt4_path = self.find_mt4_installation()
            
        if not self.mt4_path:
            self.log("無法找到 MT4 安裝路徑", 'error')
            self.log("請使用 --mt4-path 參數指定 MT4 安裝路徑", 'error')
            return False
        
        self.log(f"使用 MT4 路徑: {self.mt4_path}", 'success')
        
        # 2. 建立各種目錄
        steps = [
            ("基本目錄", self.create_basic_directories),
            ("橋接目錄", self.create_bridge_directories),
            ("ZeroMQ 目錄", self.create_zeromq_directories),
            ("Python 橋接目錄", self.create_python_bridge_directories),
            ("設定檔案", self.create_configuration_files),
            ("說明檔案", self.create_readme_files),
            ("目錄權限", self.set_directory_permissions)
        ]
        
        overall_success = True
        
        for step_name, step_function in steps:
            try:
                step_success = step_function()
                if not step_success:
                    overall_success = False
                    self.log(f"{step_name} 設置未完全成功", 'warning')
            except Exception as e:
                overall_success = False
                self.log(f"執行 {step_name} 時發生錯誤: {str(e)}", 'error')
        
        # 3. 生成報告
        report = self.generate_summary_report()
        self.save_report(report)
        
        # 4. 顯示摘要
        self.print_final_summary(report)
        
        return overall_success
    
    def print_final_summary(self, report: Dict):
        """顯示最終摘要"""
        print("\n" + "=" * 60)
        print("📊 資料夾準備完成摘要")
        print("=" * 60)
        
        status_map = {
            'SUCCESS': ('🟢', '完全成功'),
            'PARTIAL': ('🟡', '部分成功'),
            'FAILED': ('🔴', '執行失敗')
        }
        
        status_color, status_text = status_map.get(report['summary']['overall_status'], ('❓', '未知狀態'))
        
        print(f"{status_color} 總體狀態: {status_text}")
        print(f"📁 已建立資料夾: {report['summary']['total_folders_created']}")
        print(f"📄 已建立檔案: {report['summary']['total_files_created']}")
        print(f"❌ 錯誤數量: {report['summary']['errors_count']}")
        print(f"⚠️  警告數量: {report['summary']['warnings_count']}")
        
        if report['created_folders']:
            print(f"\n主要建立的資料夾:")
            for folder in report['created_folders'][:5]:  # 顯示前5個
                relative_path = os.path.relpath(folder, self.mt4_path)
                print(f"  • {relative_path}")
            
            if len(report['created_folders']) > 5:
                print(f"  ... 以及其他 {len(report['created_folders']) - 5} 個資料夾")
        
        if report['summary']['errors_count'] > 0:
            print(f"\n需要注意的錯誤:")
            for error in report['errors'][:3]:
                print(f"  • {error}")
        
        print(f"\n📄 詳細報告: mt4_folder_preparation_report.json")
        
        # 下一步建議
        if report['summary']['overall_status'] == 'SUCCESS':
            print(f"\n🎉 資料夾準備完成！")
            print(f"   下一步建議：")
            print(f"   1. 執行 check_mt4_env.py 驗證環境")
            print(f"   2. 安裝必要的 Python 套件")
            print(f"   3. 測試橋接通訊功能")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='MT4 資料夾準備腳本')
    parser.add_argument('--mt4-path', type=str, help='指定 MT4 安裝路徑')
    parser.add_argument('--force', '-f', action='store_true', help='強制重新建立已存在的目錄和檔案')
    parser.add_argument('--verbose', '-v', action='store_true', help='顯示詳細資訊')
    
    args = parser.parse_args()
    
    # 建立準備器實例
    preparer = MT4FolderPreparer(
        mt4_path=args.mt4_path,
        verbose=args.verbose,
        force=args.force
    )
    
    try:
        # 執行完整準備
        success = preparer.run_full_preparation()
        
        # 根據結果設定退出碼
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  準備過程已中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 執行準備時發生未預期的錯誤: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()