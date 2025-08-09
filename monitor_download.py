"""
Monitor download progress and report when complete
"""
import json
import time
import os
from datetime import datetime

def check_progress():
    """Check download progress from status files"""
    status_file = 'scripts/download/download_status.json'
    checkpoint_file = 'scripts/download/download_checkpoint.json'
    
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            status = json.load(f)
            
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                completed = len(checkpoint.get('completed', []))
        else:
            completed = 0
            
        print(f"\n{'='*60}")
        print(f"Download Progress Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        print(f"Progress: {status.get('progress', 'N/A')}")
        print(f"Completed: {completed} stocks")
        print(f"Records Downloaded: {status.get('records_downloaded', 0):,}")
        print(f"Storage Used: {status.get('storage_used_mb', 0):.1f} MB")
        print(f"Elapsed Time: {status.get('elapsed_time', 'N/A')}")
        print(f"Estimated Completion: {status.get('estimated_completion', 'N/A')}")
        print(f"Failed Count: {status.get('failed_count', 0)}")
        
        # Check if complete
        if completed >= 4215:
            print(f"\n✅ DOWNLOAD COMPLETE!")
            return True
        else:
            remaining = 4215 - completed
            print(f"\n⏳ Still downloading... {remaining} stocks remaining")
            return False
    else:
        print("Status file not found. Download may not be running.")
        return False

def main():
    """Monitor until complete"""
    print("Starting download monitor...")
    print("Will check progress every 30 seconds")
    
    while True:
        if check_progress():
            print("\n🎉 All data downloaded successfully!")
            break
        
        # Wait 30 seconds before next check
        time.sleep(30)

if __name__ == "__main__":
    main()