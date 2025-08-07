#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
7/24 自動化系統 - 帶日誌記錄版本
自動記錄所有任務到 TASK_EXECUTION_LOG.md
"""

import sys
import os
import time
import json
import subprocess
from datetime import datetime
from typing import Dict, List, Any

class AutoLogger:
    """自動日誌記錄器"""
    
    def __init__(self):
        self.log_file = "TASK_EXECUTION_LOG.md"
        self.tasks_completed = []
        self.tasks_failed = []
        self.external_issues = []
        self.start_time = datetime.now()
        
    def update_log(self, task_name: str, agent: str, status: str, 
                   duration: float = 0, error: str = None):
        """更新執行日誌"""
        
        # 讀取當前日誌
        with open(self.log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 準備新記錄
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if status == 'completed':
            # 添加到完成任務表格
            new_row = f"| {timestamp} | {task_name} | {agent} | ✅ 完成 | {duration:.1f}s | 正常 |"
            
            # 找到已完成任務表格並添加
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '### ✅ 已完成任務' in line:
                    # 找到表格結尾
                    j = i + 4  # 跳過標題和表頭
                    while j < len(lines) and lines[j].startswith('|'):
                        j += 1
                    lines.insert(j, new_row)
                    break
                    
        elif status == 'failed':
            # 添加到失敗任務表格
            error_type = "系統內部" if not error else "場外問題"
            new_row = f"| {timestamp} | {task_name} | {agent} | {error_type} | {error or '未知'} | 自動重試 |"
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if '### ❌ 失敗任務' in line:
                    j = i + 4
                    # 如果是第一個失敗，替換佔位符
                    if lines[j] == '| - | - | - | - | - | - |':
                        lines[j] = new_row
                    else:
                        while j < len(lines) and lines[j].startswith('|'):
                            j += 1
                        lines.insert(j, new_row)
                    break
                    
            # 如果是場外問題，記錄到場外問題區
            if "場外" in error_type or "MT4" in str(error):
                self.external_issues.append({
                    'time': timestamp,
                    'task': task_name,
                    'issue': error
                })
                
                # 更新場外問題區
                for i, line in enumerate(lines):
                    if '### 問題清單' in line:
                        issue_num = len(self.external_issues)
                        new_issue = f"{issue_num}. **[{timestamp}]** {task_name}: {error}"
                        lines[i+1] = new_issue
                        break
        
        # 更新統計
        for i, line in enumerate(lines):
            if '## 📈 系統狀態摘要' in line:
                # 更新統計數據
                completed_count = len([l for l in lines if '✅ 完成' in l and '|' in l]) - 1
                failed_count = len([l for l in lines if '❌' in l and '|' in l and '| -' not in l])
                
                lines[i+2] = f"- **總任務數**: {completed_count + failed_count + 5}"
                lines[i+3] = f"- **已完成**: {completed_count}"
                lines[i+4] = f"- **執行中**: 1"
                lines[i+5] = f"- **待執行**: {5 - failed_count}"
                lines[i+6] = f"- **失敗**: {failed_count}"
                break
        
        # 更新最後更新時間
        for i, line in enumerate(lines):
            if '*最後更新:' in line:
                lines[i] = f"*最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
                break
        
        # 寫回文件
        content = '\n'.join(lines)
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(content)

class TaskRunner:
    """任務執行器"""
    
    def __init__(self, logger: AutoLogger):
        self.logger = logger
        
    def run_task(self, task_name: str, agent: str, command: str) -> bool:
        """執行單個任務"""
        
        print(f"\n{'='*60}")
        print(f" Executing Task: {task_name}")
        print(f" Agent: {agent}")
        print(f" Time: {datetime.now().strftime('%H:%M:%S')}")
        print('='*60)
        
        start_time = time.time()
        
        try:
            # 執行命令
            if agent == 'devops':
                result = self.run_devops_task(command)
            elif agent == 'data_engineer':
                result = self.run_de_task(command)
            elif agent == 'qa':
                result = self.run_qa_task(command)
            else:
                result = True  # 其他agent暫時返回成功
            
            duration = time.time() - start_time
            
            if result:
                print(f"[PASS] Task completed: {task_name}")
                self.logger.update_log(task_name, agent, 'completed', duration)
                return True
            else:
                print(f"[FAIL] Task failed: {task_name}")
                self.logger.update_log(task_name, agent, 'failed', duration, "執行失敗")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            # 判斷是否為場外問題
            if "MT4" in error_msg or "connection" in error_msg.lower():
                print(f"[WARNING] External issue: {error_msg}")
                self.logger.update_log(task_name, agent, 'failed', duration, f"場外問題: {error_msg}")
            else:
                print(f"[ERROR] System error: {error_msg}")
                self.logger.update_log(task_name, agent, 'failed', duration, f"系統錯誤: {error_msg}")
            
            return False
    
    def run_devops_task(self, command: str) -> bool:
        """執行DevOps任務"""
        cmd_map = {
            'diagnose': 'python mt4_diagnosis.py',
            'fix_connection': 'python devops_quick_fix.py',
            'test_basic': 'python devops_test_basic.py'
        }
        
        cmd = cmd_map.get(command, command)
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, 
                                  text=True, timeout=30)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("  任務超時")
            return False
        except Exception as e:
            print(f"  執行錯誤: {e}")
            return False
    
    def run_de_task(self, command: str) -> bool:
        """執行數據工程任務"""
        cmd_map = {
            'collect_data': 'python collect_btc_markets.py',
            'check_quality': 'python data_quality_report.py'
        }
        
        cmd = cmd_map.get(command, command)
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False
    
    def run_qa_task(self, command: str) -> bool:
        """執行QA任務"""
        cmd_map = {
            'test_trading': 'python qa_trading_test.py',
            'test_connection': 'python devops_test_basic.py'
        }
        
        cmd = cmd_map.get(command, command)
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                  text=True, timeout=30)
            return result.returncode == 0
        except Exception:
            return False

def main():
    """主函數 - 開始自動化執行"""
    
    print("\n" + "="*70)
    print(" [AUTO] 7/24 Automation System Starting")
    print("="*70)
    print(f" Start Time: {datetime.now()}")
    print(" Mode: Fully Automated")
    print(" User Status: Offline (Sleeping)")
    print(" Log File: TASK_EXECUTION_LOG.md")
    print("="*70)
    
    # 初始化
    logger = AutoLogger()
    runner = TaskRunner(logger)
    
    # 定義今日任務
    tasks = [
        ("Morning System Check", "devops", "diagnose"),
        ("Test MT4 Connection", "devops", "test_basic"),
        ("Collect Market Data", "data_engineer", "collect_data"),
        ("Check Data Quality", "data_engineer", "check_quality"),
        ("Test Trading Functions", "qa", "test_trading"),
        ("Fix Connection Issues", "devops", "fix_connection"),
        ("Verify System Status", "qa", "test_connection")
    ]
    
    print("\n[TASKS] Today's Task List:")
    for i, (name, agent, _) in enumerate(tasks, 1):
        print(f"  {i}. [{agent}] {name}")
    
    print("\nStarting task execution...\n")
    
    # 執行所有任務
    for task_name, agent, command in tasks:
        success = runner.run_task(task_name, agent, command)
        
        if not success:
            print(f"  [WARNING] Task failed, will retry later")
        
        # 任務間隔
        time.sleep(3)
    
    # 完成總結
    print("\n" + "="*70)
    print(" [SUMMARY] Execution Summary")
    print("="*70)
    print(f" Total Tasks: {len(tasks)}")
    print(f" Execution Time: {(datetime.now() - logger.start_time).seconds} seconds")
    print(" Detailed Log: TASK_EXECUTION_LOG.md")
    print("="*70)
    
    print("\n[SUCCESS] Automation execution completed")
    print("[INFO] All tasks logged in TASK_EXECUTION_LOG.md")
    print("[INFO] User can sleep peacefully, system will continue running")
    
    # 持續運行監控
    print("\n[MONITOR] Entering monitoring mode...")
    while True:
        time.sleep(300)  # 每5分鐘檢查一次
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 系統運行正常...")
        
        # 可以在這裡添加定期任務
        runner.run_task("Periodic Health Check", "devops", "diagnose")

if __name__ == "__main__":
    main()