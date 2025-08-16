"""
Stage 0: Fix logging - Replace print statements with proper logging
"""

import os
import sys
import re
from pathlib import Path
from typing import List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def create_logger_config():
    """Create standard logger configuration"""
    config = '''"""
Standard logging configuration for the project
"""

import logging
import logging.handlers
import os
from pathlib import Path

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Setup a standard logger for the project
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # File handler
    log_dir = Path(__file__).parent.parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / 'trading_system.log',
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Create a default logger instance
default_logger = setup_logger('trading_system')
'''
    
    config_path = project_root / 'src' / 'utils' / 'logger_config.py'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config)
    
    print(f"[OK] Created logger configuration at {config_path}")
    return config_path

def fix_file_logging(file_path: Path) -> bool:
    """Replace print statements with logging in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        modified = False
        new_lines = []
        needs_import = True
        has_logger = False
        
        # Check if file already has logging
        for line in lines:
            if 'import logging' in line or 'from loguru import logger' in line:
                needs_import = False
            if 'logger = ' in line or 'self.logger' in line:
                has_logger = True
        
        # Add imports if needed
        if needs_import and not has_logger:
            # Find the right place to add imports (after initial comments/docstrings)
            import_added = False
            for i, line in enumerate(lines):
                new_lines.append(line)
                if not import_added and not line.strip().startswith('#') and not line.strip().startswith('"""'):
                    if 'import' in line or 'from' in line:
                        # Add after first import block
                        if i + 1 < len(lines) and not ('import' in lines[i + 1] or 'from' in lines[i + 1]):
                            new_lines.append('import logging\n')
                            new_lines.append('\n')
                            new_lines.append('# Setup logger\n')
                            new_lines.append('logger = logging.getLogger(__name__)\n')
                            new_lines.append('\n')
                            import_added = True
                            modified = True
        else:
            new_lines = lines.copy()
        
        # Replace print statements
        final_lines = []
        for line in new_lines:
            original_line = line
            
            # Skip comments and strings
            if line.strip().startswith('#'):
                final_lines.append(line)
                continue
            
            # Pattern replacements
            replacements = [
                (r'print\s*\(\s*f?"?\[ERROR\]([^)]+)\)', r'logger.error(\1)'),
                (r'print\s*\(\s*f?"?\[WARNING\]([^)]+)\)', r'logger.warning(\1)'),
                (r'print\s*\(\s*f?"?\[INFO\]([^)]+)\)', r'logger.info(\1)'),
                (r'print\s*\(\s*f?"?\[DEBUG\]([^)]+)\)', r'logger.debug(\1)'),
                (r'print\s*\(\s*f?"?\[OK\]([^)]+)\)', r'logger.info("✓" + \1)'),
                (r'print\s*\(\s*f?"?\[SUCCESS\]([^)]+)\)', r'logger.info("✓" + \1)'),
                (r'print\s*\(\s*f?"?\[FAILED?\]([^)]+)\)', r'logger.error("✗" + \1)'),
            ]
            
            for pattern, replacement in replacements:
                if re.search(pattern, line):
                    line = re.sub(pattern, replacement, line)
                    modified = True
            
            final_lines.append(line)
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(final_lines)
            return True
        
        return False
        
    except Exception as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    print("\n" + "="*80)
    print("STAGE 0: LOGGING STANDARDIZATION")
    print("="*80)
    
    # Create logger configuration
    print("\n[1] Creating standard logger configuration...")
    create_logger_config()
    
    # Fix specific files known to have many print statements
    print("\n[2] Fixing logging in key files...")
    
    key_files = [
        'src/connectors/capital_com_api.py',
        'src/capital_service.py',
        'data_pipeline/free_data_client.py',
        'config/config.py',
        'src/core/trading_system.py',
    ]
    
    fixed_count = 0
    for file_path in key_files:
        full_path = project_root / file_path
        if full_path.exists():
            if fix_file_logging(full_path):
                print(f"  [FIXED] {file_path}")
                fixed_count += 1
            else:
                print(f"  [SKIP] {file_path} - already has logging or no changes needed")
    
    print(f"\n[OK] Fixed logging in {fixed_count} files")
    
    # Create a sample usage file
    sample = '''"""
Example of proper logging usage
"""

import logging
from src.utils.logger_config import setup_logger

# Setup logger for this module
logger = setup_logger(__name__)

def example_function():
    """Example function showing logging usage"""
    
    # Different log levels
    logger.debug("Debug message - detailed diagnostic info")
    logger.info("Info message - general informational messages")
    logger.warning("Warning message - something unexpected happened")
    logger.error("Error message - a serious problem occurred")
    logger.critical("Critical message - the program may not be able to continue")
    
    # With formatting
    value = 42
    logger.info(f"Processing value: {value}")
    
    # Exception logging
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.error(f"Division error occurred: {e}", exc_info=True)
    
    # Success/failure patterns
    logger.info("✓ Operation completed successfully")
    logger.error("✗ Operation failed")

if __name__ == "__main__":
    example_function()
'''
    
    example_path = project_root / 'examples' / 'logging_example.py'
    example_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(sample)
    
    print(f"\n[3] Created logging example at {example_path}")
    
    print("\n" + "="*60)
    print("LOGGING STANDARDIZATION COMPLETE")
    print("="*60)
    
    print("\nNext steps:")
    print("1. Use 'from src.utils.logger_config import setup_logger'")
    print("2. Create logger: logger = setup_logger(__name__)")
    print("3. Replace print() with logger.info(), logger.error(), etc.")
    print("4. Check logs in 'logs/trading_system.log'")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)