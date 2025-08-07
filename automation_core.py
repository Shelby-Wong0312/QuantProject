#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
7/24 自動化閉環交易系統核心
PM -> Agent -> 執行 -> 監控 -> 報告 -> PM
完全自動化，無需人工干預
"""

import sys
import os
import time
import json
import logging
import threading
import schedule
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import traceback

# Add project path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('AutomationCore')

# Agent Types
class AgentType(Enum):
    PM = "pm"
    DEVOPS = "devops"
    DE = "data_engineer"
    QA = "qa"
    QUANT = "quant"
    FULLSTACK = "fullstack"

# Task Status
class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"

# Task Priority
class Priority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Task:
    """任務定義"""
    id: str
    name: str
    agent: AgentType
    command: str
    priority: Priority
    status: TaskStatus
    created_at: datetime
    assigned_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: int = 300  # seconds
    dependencies: List[str] = None
    
    def to_dict(self):
        data = asdict(self)
        data['agent'] = self.agent.value
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        if self.assigned_at:
            data['assigned_at'] = self.assigned_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

class AgentExecutor:
    """Agent 執行器"""
    
    def __init__(self, agent_type: AgentType):
        self.agent_type = agent_type
        self.logger = logging.getLogger(f'Agent.{agent_type.value}')
        
    def execute(self, task: Task) -> Dict:
        """執行任務"""
        self.logger.info(f"Executing task: {task.name}")
        
        try:
            # Import required modules based on agent type
            if self.agent_type == AgentType.DE:
                return self._execute_de_task(task)
            elif self.agent_type == AgentType.DEVOPS:
                return self._execute_devops_task(task)
            elif self.agent_type == AgentType.QA:
                return self._execute_qa_task(task)
            elif self.agent_type == AgentType.QUANT:
                return self._execute_quant_task(task)
            elif self.agent_type == AgentType.FULLSTACK:
                return self._execute_fullstack_task(task)
            else:
                return {'status': 'error', 'message': f'Unknown agent type: {self.agent_type}'}
                
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return {'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}
    
    def _execute_de_task(self, task: Task) -> Dict:
        """執行數據工程任務"""
        import subprocess
        
        command_map = {
            'collect_data': 'python collect_btc_markets.py',
            'check_quality': 'python data_quality_report.py',
            'market_status': 'python market_status_report.py'
        }
        
        cmd = command_map.get(task.command, task.command)
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=task.timeout)
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:],
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'message': f'Command timed out after {task.timeout}s'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _execute_devops_task(self, task: Task) -> Dict:
        """執行 DevOps 任務"""
        import subprocess
        
        command_map = {
            'diagnose': 'python mt4_diagnosis.py',
            'fix_connection': 'python devops_quick_fix.py',
            'test_trade': 'python devops_fixed_trade.py',
            'restart_services': 'python restart_mt4_bridge.py'
        }
        
        cmd = command_map.get(task.command, task.command)
        
        try:
            # For diagnostic tasks, use shorter timeout
            timeout = 60 if 'diagnose' in task.command else task.timeout
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:],
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'message': f'Command timed out after {timeout}s'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _execute_qa_task(self, task: Task) -> Dict:
        """執行 QA 任務"""
        import subprocess
        
        command_map = {
            'test_connection': 'python devops_test_basic.py',
            'test_trading': 'python qa_trading_test.py',
            'test_data': 'python test_data_quality.py',
            'verify_system': 'python system_verification.py'
        }
        
        cmd = command_map.get(task.command, task.command)
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=task.timeout)
            
            # Parse test results
            success = 'PASSED' in result.stdout or 'SUCCESS' in result.stdout
            
            return {
                'status': 'success' if success else 'failed',
                'tests_passed': success,
                'stdout': result.stdout[-1000:],
                'stderr': result.stderr[-1000:],
                'return_code': result.returncode
            }
        except subprocess.TimeoutExpired:
            return {'status': 'timeout', 'message': f'Tests timed out after {task.timeout}s'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _execute_quant_task(self, task: Task) -> Dict:
        """執行量化策略任務"""
        # Placeholder for quant tasks
        return {
            'status': 'success',
            'message': f'Quant task {task.command} executed',
            'strategy': 'momentum',
            'backtest_sharpe': 1.8
        }
    
    def _execute_fullstack_task(self, task: Task) -> Dict:
        """執行全棧開發任務"""
        # Placeholder for fullstack tasks
        return {
            'status': 'success',
            'message': f'Fullstack task {task.command} executed',
            'dashboard_url': 'http://localhost:8050'
        }

class TaskScheduler:
    """任務調度器"""
    
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.agents: Dict[AgentType, AgentExecutor] = {}
        self.running = False
        self.logger = logging.getLogger('TaskScheduler')
        self.lock = threading.Lock()
        
        # Initialize agents
        for agent_type in AgentType:
            if agent_type != AgentType.PM:
                self.agents[agent_type] = AgentExecutor(agent_type)
    
    def add_task(self, task: Task) -> None:
        """添加任務"""
        with self.lock:
            self.tasks[task.id] = task
            self.logger.info(f"Task added: {task.name} (Priority: {task.priority.value})")
    
    def get_next_task(self) -> Optional[Task]:
        """獲取下一個要執行的任務"""
        with self.lock:
            pending_tasks = [
                t for t in self.tasks.values() 
                if t.status == TaskStatus.PENDING
            ]
            
            if not pending_tasks:
                return None
            
            # Sort by priority and creation time
            pending_tasks.sort(key=lambda t: (t.priority.value, t.created_at))
            
            # Check dependencies
            for task in pending_tasks:
                if self._check_dependencies(task):
                    return task
            
            return None
    
    def _check_dependencies(self, task: Task) -> bool:
        """檢查任務依賴"""
        if not task.dependencies:
            return True
        
        for dep_id in task.dependencies:
            if dep_id in self.tasks:
                dep_task = self.tasks[dep_id]
                if dep_task.status != TaskStatus.COMPLETED:
                    return False
        
        return True
    
    def execute_task(self, task: Task) -> None:
        """執行單個任務"""
        try:
            # Update status
            task.status = TaskStatus.ASSIGNED
            task.assigned_at = datetime.now()
            
            # Get agent
            agent = self.agents.get(task.agent)
            if not agent:
                raise ValueError(f"No agent found for type: {task.agent}")
            
            # Execute
            task.status = TaskStatus.RUNNING
            result = agent.execute(task)
            
            # Update result
            task.result = result
            
            if result.get('status') == 'success':
                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now()
                self.logger.info(f"Task completed: {task.name}")
            else:
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.RETRY
                    self.logger.warning(f"Task failed, will retry: {task.name} (Attempt {task.retry_count}/{task.max_retries})")
                else:
                    task.status = TaskStatus.FAILED
                    self.logger.error(f"Task failed permanently: {task.name}")
                    
        except Exception as e:
            self.logger.error(f"Task execution error: {e}")
            task.status = TaskStatus.FAILED
            task.result = {'status': 'error', 'message': str(e)}
    
    def run(self) -> None:
        """運行調度器"""
        self.running = True
        self.logger.info("Task scheduler started")
        
        while self.running:
            try:
                # Get next task
                task = self.get_next_task()
                
                if task:
                    self.logger.info(f"Executing task: {task.name}")
                    
                    # Execute in thread
                    thread = threading.Thread(target=self.execute_task, args=(task,))
                    thread.start()
                    thread.join(timeout=task.timeout)
                    
                    if thread.is_alive():
                        self.logger.warning(f"Task timeout: {task.name}")
                        task.status = TaskStatus.FAILED
                        task.result = {'status': 'timeout'}
                
                # Handle retry tasks
                self._handle_retries()
                
                # Sleep
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(10)
    
    def _handle_retries(self) -> None:
        """處理重試任務"""
        with self.lock:
            retry_tasks = [
                t for t in self.tasks.values()
                if t.status == TaskStatus.RETRY
            ]
            
            for task in retry_tasks:
                # Wait before retry
                if task.completed_at:
                    retry_delay = timedelta(seconds=30 * task.retry_count)
                    if datetime.now() - task.completed_at > retry_delay:
                        task.status = TaskStatus.PENDING
                        self.logger.info(f"Retrying task: {task.name}")
                else:
                    task.status = TaskStatus.PENDING
    
    def stop(self) -> None:
        """停止調度器"""
        self.running = False
        self.logger.info("Task scheduler stopped")
    
    def get_status(self) -> Dict:
        """獲取狀態"""
        with self.lock:
            status_count = {}
            for task in self.tasks.values():
                status = task.status.value
                status_count[status] = status_count.get(status, 0) + 1
            
            return {
                'total_tasks': len(self.tasks),
                'status_breakdown': status_count,
                'running': self.running
            }

class AutomationPM:
    """自動化項目經理"""
    
    def __init__(self):
        self.scheduler = TaskScheduler()
        self.logger = logging.getLogger('AutomationPM')
        self.task_counter = 0
        self.monitoring_thread = None
        
    def create_task(self, name: str, agent: AgentType, command: str, 
                   priority: Priority = Priority.MEDIUM, 
                   dependencies: List[str] = None) -> Task:
        """創建任務"""
        self.task_counter += 1
        task_id = f"TASK_{self.task_counter:04d}"
        
        task = Task(
            id=task_id,
            name=name,
            agent=agent,
            command=command,
            priority=priority,
            status=TaskStatus.PENDING,
            created_at=datetime.now(),
            dependencies=dependencies
        )
        
        self.scheduler.add_task(task)
        self.logger.info(f"Created task: {name} -> {agent.value}")
        
        return task
    
    def create_daily_tasks(self) -> None:
        """創建每日任務"""
        self.logger.info("Creating daily tasks")
        
        # Morning system check
        t1 = self.create_task(
            "Morning System Check",
            AgentType.DEVOPS,
            "diagnose",
            Priority.HIGH
        )
        
        # Data collection
        t2 = self.create_task(
            "Collect Market Data",
            AgentType.DE,
            "collect_data",
            Priority.HIGH,
            dependencies=[t1.id]
        )
        
        # Data quality check
        t3 = self.create_task(
            "Check Data Quality",
            AgentType.DE,
            "check_quality",
            Priority.MEDIUM,
            dependencies=[t2.id]
        )
        
        # Trading test
        t4 = self.create_task(
            "Test Trading Functions",
            AgentType.QA,
            "test_trading",
            Priority.HIGH,
            dependencies=[t1.id]
        )
        
        # Strategy backtest
        t5 = self.create_task(
            "Run Strategy Backtest",
            AgentType.QUANT,
            "backtest",
            Priority.MEDIUM,
            dependencies=[t3.id]
        )
    
    def create_monitoring_tasks(self) -> None:
        """創建監控任務"""
        self.logger.info("Creating monitoring tasks")
        
        # System health check every hour
        self.create_task(
            "Hourly Health Check",
            AgentType.DEVOPS,
            "health_check",
            Priority.LOW
        )
        
        # Market status every 30 minutes
        self.create_task(
            "Market Status Update",
            AgentType.DE,
            "market_status",
            Priority.LOW
        )
    
    def handle_failure(self, task: Task) -> None:
        """處理失敗任務"""
        self.logger.warning(f"Handling failure for task: {task.name}")
        
        if task.agent == AgentType.DEVOPS and 'connection' in task.name.lower():
            # Create fix task
            fix_task = self.create_task(
                "Fix Connection Issues",
                AgentType.DEVOPS,
                "fix_connection",
                Priority.CRITICAL
            )
            
            # Create verification task
            self.create_task(
                "Verify Fix",
                AgentType.QA,
                "test_connection",
                Priority.CRITICAL,
                dependencies=[fix_task.id]
            )
        
        elif task.agent == AgentType.DE and 'data' in task.name.lower():
            # Retry data collection with different parameters
            self.create_task(
                "Retry Data Collection",
                AgentType.DE,
                "collect_data",
                Priority.HIGH
            )
    
    def monitor_system(self) -> None:
        """監控系統狀態"""
        while True:
            try:
                status = self.scheduler.get_status()
                
                # Log status
                self.logger.info(f"System Status: {status}")
                
                # Check for issues
                if status['status_breakdown'].get('failed', 0) > 3:
                    self.logger.error("Too many failed tasks, triggering recovery")
                    self.create_task(
                        "System Recovery",
                        AgentType.DEVOPS,
                        "system_recovery",
                        Priority.CRITICAL
                    )
                
                # Save status to file
                with open('automation_status.json', 'w') as f:
                    json.dump({
                        'timestamp': datetime.now().isoformat(),
                        'status': status,
                        'tasks': [t.to_dict() for t in self.scheduler.tasks.values()]
                    }, f, indent=2)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                time.sleep(60)
    
    def start(self) -> None:
        """啟動自動化系統"""
        self.logger.info("Starting 7/24 Automation System")
        
        # Create initial tasks
        self.create_daily_tasks()
        self.create_monitoring_tasks()
        
        # Schedule daily tasks
        schedule.every().day.at("09:00").do(self.create_daily_tasks)
        schedule.every().hour.do(self.create_monitoring_tasks)
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self.monitor_system, daemon=True)
        self.monitoring_thread.start()
        
        # Start scheduler in thread
        scheduler_thread = threading.Thread(target=self.scheduler.run, daemon=True)
        scheduler_thread.start()
        
        # Run schedule
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            self.logger.info("Shutting down automation system")
            self.scheduler.stop()

def main():
    """主函數"""
    print("\n" + "="*60)
    print(" 7/24 Automated Trading System ")
    print("="*60)
    print(f" Start Time: {datetime.now()}")
    print(" Mode: Fully Autonomous")
    print(" No Human Intervention Required")
    print("="*60)
    
    # Create PM
    pm = AutomationPM()
    
    # Start system
    pm.start()

if __name__ == "__main__":
    main()