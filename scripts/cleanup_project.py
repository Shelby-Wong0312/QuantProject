"""
清理專案中的重複和臨時文件
"""

import os
import shutil


def cleanup_project():
    """清理專案文件"""

    # 要刪除的臨時和測試文件
    files_to_delete = [
        # 測試和驗證相關的臨時文件
        "test_capital_tick_data.py",
        "test_dukascopy_direct.py",
        "test_known_stocks.py",
        "test_stock_validation.py",
        "test_tick_data_sources.py",
        "quick_stock_test.py",
        "test_data_range.py",
        "test_data_years.py",
        "test_download_sample.py",
        # 重複的驗證腳本
        "validate_all_stocks.py",
        "fast_validate_all.py",
        "continue_validation.py",
        "stock_validator_and_collector.py",
        "stock_data_collector.py",
        "comprehensive_stock_collector.py",
        "capital_stock_downloader.py",
        # 重複的下載腳本
        "download_minute_data.py",
        "download_tick_data.py",
        "get_free_tick_data.py",
        "setup_stock_data.py",
        "final_report_and_download.py",
        # 臨時和測試文件
        "cr.py",
        "nul",
        # 舊的檢查點文件
        "validation_checkpoint.pkl",
        # 重複的報告顯示文件
        "show_validation_results.py",
        # 臨時監控文件
        "monitor_download.py",
        "quick_status.py",
        "check_db_status.py",
        "show_data_storage.py",
        "restart_report_system.py",
        "auto_restart_system.py",
    ]

    # 要保留的重要文件
    important_files = [
        "start_full_download.py",  # 主要下載腳本
        "setup_sqlite_database.py",  # 數據庫設置
        "batch_validate.py",  # 批量驗證腳本
        "capital_automation_system.py",  # 自動化系統
        "src/capital_service.py",  # 核心服務
        "README.md",
        ".env",
        "requirements.txt",
    ]

    deleted_count = 0
    for file in files_to_delete:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"Could not delete {file}: {e}")

    # 清理空目錄
    dirs_to_check = ["dukascopy_python-4.0.1"]
    for dir_name in dirs_to_check:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            try:
                shutil.rmtree(dir_name)
                print(f"Deleted directory: {dir_name}")
                deleted_count += 1
            except Exception as e:
                print(f"Could not delete directory {dir_name}: {e}")

    print(f"\nCleaned up {deleted_count} files/directories")

    # 列出保留的關鍵文件
    print("\nImportant files kept:")
    for file in important_files:
        if os.path.exists(file):
            print(f"  - {file}")

    return deleted_count


if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT CLEANUP")
    print("=" * 60)
    cleanup_project()
    print("=" * 60)
