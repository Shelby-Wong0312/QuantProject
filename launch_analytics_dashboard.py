"""
QuantTrading Analytics Dashboard Launcher

Quick launcher for the analytics dashboard with system checks and setup.
"""

import os
import sys
import subprocess
import sqlite3
from datetime import datetime

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy',
        'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} - OK")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Missing")
    
    if missing_packages:
        print(f"\\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("\\n✅ All dependencies are installed!")
    return True

def check_database():
    """Check if trading database exists and has data"""
    db_paths = [
        "data/live_trades.db",
        "data/live_trades_full.db", 
        "demo_analytics.db"
    ]
    
    for db_path in db_paths:
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check if trades table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trades';")
                if cursor.fetchone():
                    # Check for data
                    cursor.execute("SELECT COUNT(*) FROM trades;")
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        print(f"✅ Database found: {db_path} ({count} trades)")
                        conn.close()
                        return db_path
                    else:
                        print(f"⚠️  Database found but empty: {db_path}")
                else:
                    print(f"⚠️  Database found but no trades table: {db_path}")
                
                conn.close()
                
            except Exception as e:
                print(f"❌ Error checking database {db_path}: {e}")
        else:
            print(f"❌ Database not found: {db_path}")
    
    print("\\n⚠️  No valid trading database found!")
    return None

def create_demo_data():
    """Create demo data if no database exists"""
    print("\\n🎲 Creating demo trading data...")
    
    try:
        # Import and run demo data creation
        sys.path.append('src')
        from analytics.demo_analytics import AnalyticsDemo
        
        demo = AnalyticsDemo("demo_analytics.db")
        trades_count = demo.create_sample_database()
        
        print(f"✅ Created demo database with {trades_count} trades")
        return "demo_analytics.db"
        
    except Exception as e:
        print(f"❌ Error creating demo data: {e}")
        return None

def launch_dashboard(db_path=None):
    """Launch the Streamlit dashboard"""
    print("\\n🚀 Launching QuantTrading Analytics Dashboard...")
    
    # Set environment variable for database path if provided
    if db_path:
        os.environ['ANALYTICS_DB_PATH'] = db_path
    
    # Launch Streamlit
    try:
        cmd = [sys.executable, "-m", "streamlit", "run", "src/analytics/streamlit_app.py"]
        
        print("\\n" + "="*60)
        print("📊 QUANTTRADING ANALYTICS DASHBOARD")
        print("="*60)
        print(f"🌐 Dashboard will be available at: http://localhost:8501")
        print("🔧 Using database:", db_path if db_path else "default")
        print("⚠️  Press Ctrl+C to stop the dashboard")
        print("="*60)
        
        # Run Streamlit
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\\n\\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"\\n❌ Error launching dashboard: {e}")

def main():
    """Main launcher function"""
    print("🚀 QuantTrading Analytics Dashboard Launcher")
    print("=" * 50)
    
    # Step 1: Check dependencies
    print("\\n1️⃣ Checking dependencies...")
    if not check_dependencies():
        print("\\n❌ Please install missing dependencies and try again")
        return
    
    # Step 2: Check for database
    print("\\n2️⃣ Checking for trading database...")
    db_path = check_database()
    
    # Step 3: Create demo data if needed
    if not db_path:
        response = input("\\n❓ No trading data found. Create demo data? (y/n): ").lower()
        if response == 'y':
            db_path = create_demo_data()
        else:
            print("\\n⚠️  Dashboard will run with empty data")
    
    # Step 4: Launch dashboard
    launch_dashboard(db_path)

if __name__ == "__main__":
    main()