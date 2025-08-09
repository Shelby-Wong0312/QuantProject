"""
Keep download running continuously until complete
"""
import subprocess
import json
import time
import os
from datetime import datetime

def is_download_complete():
    """Check if download is complete"""
    checkpoint_file = 'scripts/download/download_checkpoint.json'
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            completed = len(checkpoint.get('completed', []))
            return completed >= 4215
    return False

def is_download_running():
    """Check if download process is running"""
    try:
        # Check if python process is running with start_full_download.py
        result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'], 
                              capture_output=True, text=True)
        return 'python.exe' in result.stdout
    except:
        return False

def start_download():
    """Start the download process"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting download process...")
    subprocess.Popen(['python', 'scripts/download/start_full_download.py'], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE)
    time.sleep(5)  # Wait for process to start

def main():
    """Main loop to keep download running"""
    print("=" * 60)
    print("AUTO DOWNLOAD KEEPER")
    print("=" * 60)
    print("This script will ensure the download completes")
    print("even if the process stops unexpectedly.")
    print("Press Ctrl+C to stop.")
    print("=" * 60)
    
    check_interval = 60  # Check every minute
    
    while not is_download_complete():
        if not is_download_running():
            print(f"\n⚠️  Download process not running!")
            start_download()
        else:
            # Read current status
            status_file = 'scripts/download/download_status.json'
            if os.path.exists(status_file):
                with open(status_file, 'r') as f:
                    status = json.load(f)
                    print(f"\r✅ Running - {status.get('progress', 'N/A')} - "
                          f"Records: {status.get('records_downloaded', 0):,} - "
                          f"ETA: {status.get('estimated_completion', 'N/A')}", end='')
        
        time.sleep(check_interval)
    
    print(f"\n\n🎉 DOWNLOAD COMPLETE!")
    print(f"All 4,215 stocks have been downloaded successfully!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopped by user.")