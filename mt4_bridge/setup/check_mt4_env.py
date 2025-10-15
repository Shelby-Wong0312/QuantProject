#!/usr/bin/env python3
"""
MT4 環境檢查腳本 (check_mt4_env.py)

此腳本用於檢查 MetaTrader 4 是否已正確安裝，驗證必要的目錄結構，
並測試 Demo 帳戶連接狀態。

使用方法:
    python check_mt4_env.py [--verbose] [--fix-directories]

選項:
    --verbose           顯示詳細的檢查資訊
    --fix-directories   自動修復缺失的目錄結構
"""

import os
import sys
import subprocess
import platform
import argparse
import socket
from typing import List, Dict
import json
import time


class MT4EnvironmentChecker:
    """MT4 環境檢查器"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.system = platform.system()
        self.errors = []
        self.warnings = []
        self.info = []

        # MT4 常見安裝路徑
        self.common_mt4_paths = self._get_common_mt4_paths()

        # 必要的目錄結構
        self.required_directories = [
            "MQL4",
            "MQL4/Experts",
            "MQL4/Indicators",
            "MQL4/Scripts",
            "MQL4/Include",
            "MQL4/Libraries",
            "MQL4/Files",
            "Profiles",
            "Templates",
            "Tester",
        ]

        # Capital.com 伺服器資訊
        self.capital_servers = {"demo": "Capital.com-Demo", "real": "Capital.com-Real"}

    def _get_common_mt4_paths(self) -> List[str]:
        """取得 MT4 常見安裝路徑"""
        paths = []

        if self.system == "Windows":
            # Windows 常見路徑
            common_locations = [
                r"C:\Program Files\MetaTrader 4",
                r"C:\Program Files (x86)\MetaTrader 4",
                r"C:\Program Files\MetaTrader 4 - Capital.com",
                r"C:\Program Files (x86)\MetaTrader 4 - Capital.com",
                os.path.expanduser(r"~\AppData\Roaming\MetaQuotes\Terminal"),
            ]

            # 檢查登錄檔中的安裝路徑
            registry_paths = self._get_mt4_from_registry()
            paths.extend(registry_paths)

            # 檢查常見位置
            for path in common_locations:
                if os.path.exists(path):
                    paths.append(path)

        elif self.system == "Darwin":  # macOS
            common_locations = [
                "/Applications/MetaTrader 4.app",
                os.path.expanduser("~/Applications/MetaTrader 4.app"),
                os.path.expanduser("~/Library/Application Support/MetaQuotes/Terminal"),
            ]

            for path in common_locations:
                if os.path.exists(path):
                    paths.append(path)

        return list(set(paths))  # 去除重複

    def _get_mt4_from_registry(self) -> List[str]:
        """從 Windows 登錄檔中查找 MT4 安裝路徑"""
        paths = []

        if self.system != "Windows":
            return paths

        try:
            import winreg

            registry_keys = [
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\MetaQuotes\MetaTrader 4"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\MetaQuotes\MetaTrader 4"),
                (
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\WOW6432Node\MetaQuotes\MetaTrader 4",
                ),
            ]

            for hkey, subkey in registry_keys:
                try:
                    with winreg.OpenKey(hkey, subkey) as key:
                        try:
                            install_path, _ = winreg.QueryValueEx(key, "InstallPath")
                            if install_path and os.path.exists(install_path):
                                paths.append(install_path)
                        except FileNotFoundError:
                            pass
                except FileNotFoundError:
                    continue

        except ImportError:
            pass  # winreg 不可用

        return paths

    def log(self, message: str, level: str = "info"):
        """記錄訊息"""
        if level == "error":
            self.errors.append(message)
            print(f"❌ 錯誤: {message}")
        elif level == "warning":
            self.warnings.append(message)
            print(f"⚠️  警告: {message}")
        elif level == "info":
            self.info.append(message)
            if self.verbose:
                print(f"ℹ️  資訊: {message}")
        elif level == "success":
            print(f"✅ {message}")

    def check_mt4_installation(self) -> bool:
        """檢查 MT4 是否已安裝"""
        print("\n🔍 檢查 MT4 安裝狀態...")

        if not self.common_mt4_paths:
            self.log("未發現 MT4 安裝路徑", "error")
            self.log("請確認 MT4 已正確安裝", "error")
            return False

        for path in self.common_mt4_paths:
            if os.path.exists(path):
                self.log(f"發現 MT4 安裝於: {path}", "success")

                # 檢查是否為有效的 MT4 目錄
                if self._is_valid_mt4_directory(path):
                    self.mt4_path = path
                    return True
                else:
                    self.log(f"目錄 {path} 不是有效的 MT4 安裝目錄", "warning")

        self.log("未找到有效的 MT4 安裝", "error")
        return False

    def _is_valid_mt4_directory(self, path: str) -> bool:
        """檢查目錄是否為有效的 MT4 安裝目錄"""
        required_files = []

        if self.system == "Windows":
            required_files = ["terminal.exe", "metaeditor.exe"]
        elif self.system == "Darwin":
            required_files = ["MetaTrader 4"]

        for file in required_files:
            if not os.path.exists(os.path.join(path, file)):
                self.log(f"缺少必要檔案: {file}", "info")
                return False

        return True

    def check_directory_structure(self) -> bool:
        """檢查 MT4 目錄結構"""
        print("\n📁 檢查目錄結構...")

        if not hasattr(self, "mt4_path"):
            self.log("無法檢查目錄結構 - MT4 路徑未知", "error")
            return False

        all_good = True

        for directory in self.required_directories:
            dir_path = os.path.join(self.mt4_path, directory)

            if os.path.exists(dir_path):
                self.log(f"目錄存在: {directory}", "success")
            else:
                self.log(f"目錄缺失: {directory}", "warning")
                all_good = False

        return all_good

    def fix_directory_structure(self):
        """修復缺失的目錄結構"""
        print("\n🔧 修復目錄結構...")

        if not hasattr(self, "mt4_path"):
            self.log("無法修復目錄結構 - MT4 路徑未知", "error")
            return

        for directory in self.required_directories:
            dir_path = os.path.join(self.mt4_path, directory)

            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    self.log(f"已建立目錄: {directory}", "success")
                except PermissionError:
                    self.log(f"權限不足，無法建立目錄: {directory}", "error")
                except Exception as e:
                    self.log(f"建立目錄 {directory} 時發生錯誤: {str(e)}", "error")

    def check_capital_servers(self) -> bool:
        """檢查 Capital.com 伺服器連線"""
        print("\n🌐 檢查 Capital.com 伺服器連線...")

        # MT4 通常使用 443 port 進行連線
        server_hosts = [
            "capital.com",
            "mt4.capital.com",
            "real.capital.com",
            "demo.capital.com",
        ]

        connection_results = {}

        for host in server_hosts:
            try:
                # 測試 DNS 解析
                socket.gethostbyname(host)

                # 測試連線
                sock = socket.create_connection((host, 443), timeout=5)
                sock.close()

                connection_results[host] = True
                self.log(f"伺服器連線正常: {host}", "success")

            except socket.gaierror:
                connection_results[host] = False
                self.log(f"DNS 解析失敗: {host}", "warning")

            except (socket.timeout, ConnectionRefusedError, OSError):
                connection_results[host] = False
                self.log(f"無法連線到伺服器: {host}", "warning")

        # 至少一個伺服器可連線即可
        return any(connection_results.values())

    def check_mt4_processes(self) -> Dict[str, bool]:
        """檢查 MT4 相關程序是否正在執行"""
        print("\n🔄 檢查 MT4 程序狀態...")

        mt4_processes = (
            ["terminal.exe", "metaeditor.exe"]
            if self.system == "Windows"
            else ["MetaTrader 4"]
        )
        process_status = {}

        try:
            if self.system == "Windows":
                # Windows: 使用 tasklist
                result = subprocess.run(
                    ["tasklist"], capture_output=True, text=True, shell=True
                )
                running_processes = result.stdout.lower()

                for process in mt4_processes:
                    is_running = process.lower() in running_processes
                    process_status[process] = is_running

                    if is_running:
                        self.log(f"程序正在執行: {process}", "info")
                    else:
                        self.log(f"程序未執行: {process}", "info")

            elif self.system == "Darwin":
                # macOS: 使用 ps
                result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
                running_processes = result.stdout.lower()

                for process in mt4_processes:
                    is_running = process.lower() in running_processes
                    process_status[process] = is_running

                    if is_running:
                        self.log(f"程序正在執行: {process}", "info")
                    else:
                        self.log(f"程序未執行: {process}", "info")

        except Exception as e:
            self.log(f"檢查程序狀態時發生錯誤: {str(e)}", "warning")

        return process_status

    def check_bridge_requirements(self) -> bool:
        """檢查橋接通訊所需的目錄和檔案"""
        print("\n🌉 檢查橋接通訊需求...")

        if not hasattr(self, "mt4_path"):
            self.log("無法檢查橋接需求 - MT4 路徑未知", "error")
            return False

        # 檢查 MQL4/Files 目錄 (檔案橋接需要)
        files_dir = os.path.join(self.mt4_path, "MQL4", "Files")

        if os.path.exists(files_dir):
            self.log("檔案橋接目錄存在", "success")

            # 檢查讀寫權限
            try:
                test_file = os.path.join(files_dir, "bridge_test.txt")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                self.log("檔案橋接目錄可讀寫", "success")

            except Exception as e:
                self.log(f"檔案橋接目錄權限問題: {str(e)}", "warning")
                return False
        else:
            self.log("檔案橋接目錄不存在", "error")
            return False

        # 檢查 Python 環境
        python_ok = self._check_python_environment()

        return python_ok

    def _check_python_environment(self) -> bool:
        """檢查 Python 環境"""
        try:
            import sys

            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            self.log(f"Python 版本: {python_version}", "info")

            # 檢查必要的套件
            required_packages = ["zmq", "pandas", "numpy"]
            missing_packages = []

            for package in required_packages:
                try:
                    __import__(package)
                    self.log(f"套件已安裝: {package}", "success")
                except ImportError:
                    missing_packages.append(package)
                    self.log(f"缺少套件: {package}", "warning")

            if missing_packages:
                self.log(
                    f"請安裝缺少的套件: pip install {' '.join(missing_packages)}",
                    "warning",
                )
                return False

            return True

        except Exception as e:
            self.log(f"檢查 Python 環境時發生錯誤: {str(e)}", "error")
            return False

    def generate_report(self) -> Dict:
        """生成檢查報告"""
        {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system": self.system,
            "mt4_path": getattr(self, "mt4_path", None),
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "summary": {
                "errors_count": len(self.errors),
                "warnings_count": len(self.warnings),
                "overall_status": "PASS" if len(self.errors) == 0 else "FAIL",
            },
        }

        return report

    def save_report(self, report: Dict, filepath: str = None):
        """儲存檢查報告"""
        if filepath is None:
            filepath = os.path.join(
                os.path.dirname(__file__), "mt4_env_check_report.json"
            )

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            self.log(f"報告已儲存至: {filepath}", "success")

        except Exception as e:
            self.log(f"儲存報告時發生錯誤: {str(e)}", "error")

    def run_full_check(self, fix_directories: bool = False) -> bool:
        """執行完整的環境檢查"""
        print("🚀 開始 MT4 環境檢查...")
        print("=" * 50)

        # 1. 檢查 MT4 安裝
        mt4_installed = self.check_mt4_installation()

        if mt4_installed:
            # 2. 檢查目錄結構
            directories_ok = self.check_directory_structure()

            if not directories_ok and fix_directories:
                self.fix_directory_structure()
                directories_ok = self.check_directory_structure()

            # 3. 檢查伺服器連線
            self.check_capital_servers()

            # 4. 檢查程序狀態
            self.check_mt4_processes()

            # 5. 檢查橋接需求
            self.check_bridge_requirements()

        # 生成並儲存報告
        self.generate_report()
        self.save_report(report)

        # 顯示摘要
        self.print_summary(report)

        return report["summary"]["overall_status"] == "PASS"

    def print_summary(self, report: Dict):
        """顯示檢查摘要"""
        print("\n" + "=" * 50)
        print("📊 檢查摘要")
        print("=" * 50)

        status_color = "🟢" if report["summary"]["overall_status"] == "PASS" else "🔴"
        print(f"{status_color} 總體狀態: {report['summary']['overall_status']}")
        print(f"❌ 錯誤數量: {report['summary']['errors_count']}")
        print(f"⚠️  警告數量: {report['summary']['warnings_count']}")

        if report["mt4_path"]:
            print(f"📂 MT4 安裝路徑: {report['mt4_path']}")

        if report["summary"]["errors_count"] > 0:
            print("\n主要問題:")
            for error in report["errors"][:3]:  # 顯示前3個錯誤
                print(f"  • {error}")

        if report["summary"]["warnings_count"] > 0:
            print("\n注意事項:")
            for warning in report["warnings"][:3]:  # 顯示前3個警告
                print(f"  • {warning}")

        print("\n詳細報告已儲存至: mt4_env_check_report.json")


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description="MT4 環境檢查腳本")
    parser.add_argument("--verbose", "-v", action="store_true", help="顯示詳細資訊")
    parser.add_argument(
        "--fix-directories", "-", action="store_true", help="自動修復缺失的目錄"
    )

    args = parser.parse_args()

    # 建立檢查器實例
    checker = MT4EnvironmentChecker(verbose=args.verbose)

    try:
        # 執行完整檢查
        success = checker.run_full_check(fix_directories=args.fix_directories)

        # 根據結果設定退出碼
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⏹️  檢查已中斷")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 執行檢查時發生未預期的錯誤: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
