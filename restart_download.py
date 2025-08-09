"""
Restart the data download process with proper dependencies
"""

import subprocess
import sys
import os

def check_and_install_dependencies():
    """Check and install required dependencies"""
    
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    # Check for required packages
    required = ['pyarrow', 'fastparquet', 'pandas', 'numpy', 'requests']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[MISSING] {package} needs to be installed")
            missing.append(package)
    
    if missing:
        print(f"\nInstalling missing packages: {', '.join(missing)}")
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("Dependencies installed successfully!")
    else:
        print("\nAll dependencies are installed!")
    
    return True

def check_download_status():
    """Check current download status"""
    import sqlite3
    
    print("\n" + "=" * 60)
    print("CURRENT DOWNLOAD STATUS")
    print("=" * 60)
    
    conn = sqlite3.connect('data/quant_trading.db')
    cursor = conn.cursor()
    
    # Check downloaded stocks
    cursor.execute("""
        SELECT symbol, COUNT(*) as records 
        FROM daily_data 
        GROUP BY symbol
    """)
    downloaded = cursor.fetchall()
    
    print(f"Downloaded stocks: {len(downloaded)}")
    for symbol, records in downloaded[:10]:  # Show first 10
        print(f"  - {symbol}: {records} records")
    
    if len(downloaded) > 10:
        print(f"  ... and {len(downloaded) - 10} more stocks")
    
    # Check total target
    cursor.execute("SELECT COUNT(*) FROM stocks WHERE is_tradable = 1")
    total_stocks = cursor.fetchone()[0]
    
    progress = (len(downloaded) / total_stocks * 100) if total_stocks > 0 else 0
    print(f"\nProgress: {len(downloaded)}/{total_stocks} stocks ({progress:.2f}%)")
    
    conn.close()
    
    return len(downloaded), total_stocks

def restart_download():
    """Restart the download process"""
    
    print("\n" + "=" * 60)
    print("RESTARTING DOWNLOAD PROCESS")
    print("=" * 60)
    
    # Kill any existing download processes
    print("Stopping any existing download processes...")
    os.system("taskkill /F /IM python.exe /FI \"WINDOWTITLE eq start_full_download*\" 2>nul")
    
    # Start new download process
    print("Starting new download process...")
    script_path = "scripts/download/start_full_download.py"
    
    if os.path.exists(script_path):
        # Start in a new window
        subprocess.Popen(
            f'start "Stock Data Download" python {script_path}',
            shell=True
        )
        print(f"Download process started! Check the new window for progress.")
    else:
        print(f"Error: Download script not found at {script_path}")
        return False
    
    return True

def main():
    """Main restart function"""
    
    # Check dependencies
    if not check_and_install_dependencies():
        print("Failed to install dependencies")
        return
    
    # Check current status
    downloaded, total = check_download_status()
    
    if downloaded >= total:
        print("\nAll stocks have been downloaded!")
        return
    
    # Restart download
    if restart_download():
        print("\n" + "=" * 60)
        print("DOWNLOAD RESTARTED SUCCESSFULLY")
        print("=" * 60)
        print(f"Resuming from stock #{downloaded + 1}/{total}")
        print("The download will continue in the background.")
        print("Check 'logs/download.log' for detailed progress.")
    else:
        print("\nFailed to restart download process")

if __name__ == "__main__":
    main()