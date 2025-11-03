# QuantProject Directory Structure

## Root Directory Files
- `.env` - Environment configuration (API keys)
- `.env.example` - Environment configuration template
- `.gitignore` - Git ignore patterns
- `README.md` - Project documentation
- `requirements.txt` - Python dependencies
- `capital_automation_system.py` - Main automation system

## Directory Organization

### `/src` - Core Source Code (137 files)
- `capital_service.py` - Capital.com API service
- `/analysis` - Analysis tools
- `/backtesting` - Backtesting engine
- `/data_pipeline` - Data processing pipeline
- `/data_processing` - Data cleaning and validation
- `/data_providers` - External data providers
- `/execution` - Trade execution
- `/integration` - System integration
- `/ml` - Machine learning models
- `/models` - Trading models
- `/monitoring` - System monitoring
- `/optimization` - Strategy optimization
- `/performance` - Performance tracking
- `/risk` - Risk management
- `/rl` - Reinforcement learning
- `/rl_trading` - RL trading agents
- `/sensory_models` - Graph neural networks
- `/strategies` - Trading strategies
- `/testing` - Test modules
- `/visualization` - Data visualization
- `/visualizations` - Additional visualizations

### `/data` - All Data Files (24 files)
- `/csv` - CSV data files (9 files)
  - Market minute data (BTC, SPY, etc.)
  - Tradeable stocks list
- `/json` - JSON data files (9 files)
  - Validation reports
  - Market configuration
  - Test results
- `/mt4_ticks` - MT4 tick data
- `/ticks` - General tick data
- Database files:
  - `quant_trading.db`
  - `stock_data.db`
  - `stock_data_complete.db`
- Ticker lists:
  - `tradable_tickers.txt` (4,215 stocks)
  - `valid_tickers.txt`
  - `invalid_tickers.txt`

### `/historical_data` - Market Data Storage (14 files)
- `/daily` - Daily OHLC data (Parquet format)
- `/hourly` - Hourly data
- `/minute` - Minute data
- `/test` - Test data (AAPL, MSFT, etc.)

### `/config` - Configuration Files (2 files)
- `config.py` - Python configuration
- `db_config.json` - Database configuration

### `/scripts` - Utility Scripts (8 files)
- `/download` - Data download scripts
  - `start_full_download.py` - Main download script
- `/validation` - Validation scripts
  - `batch_validate.py` - Batch validation
- `setup_sqlite_database.py` - Database setup
- `setup_database.py` - Alternative DB setup
- `cleanup_project.py` - Project cleanup
- Batch files:
  - `run_automation.bat`
  - `run_validation.bat`
  - `start_download.bat`

### `/checkpoints` - Checkpoint Files (3 files)
- `batch_validation_checkpoint.json`
- `quick_validation_checkpoint.json`
- `download_checkpoint.json`

### `/logs` - Log Files (6 files)
- Validation logs
- Download logs
- System logs

### `/reports` - Generated Reports (6 files)
- Daily reports with timestamps

### `/documents` - Documentation (7 files)
- `TODO.md` - Task list
- `TODO_Capital_Integration.md`
- `TODO_MT4_Integration.md`
- `DEVOPS_FIX_GUIDE.md`
- `PM_PROGRESS_REPORT.md`
- `SYSTEM_GUIDE.md`
- Chinese documentation

### `/tests` - Test Suite (9 files)
- `/integration` - Integration tests
- `/optimization` - Optimization tests
- Unit tests for various components

### `/mt4_bridge` - MT4 Integration (43 files)
- Complete MT4 bridge implementation
- ZeroMQ connector
- MQL4 scripts
- Setup guides

### `/core` - Core Components (3 files)
- Event system
- Event loop

### `/execution` - Execution Layer (3 files)
- Broker interface
- Portfolio management

### `/strategies` - Trading Strategies (2 files)
- Strategy implementations

## Summary Statistics
- **Total Organized Files**: 259+
- **Validated Stocks**: 4,215
- **Historical Data Coverage**: 15 years
- **Database Records**: 35,208+
- **Cleanup Performed**: 
  - Deleted 29 unnecessary files
  - Organized 45 files into proper directories

## Key Features
- Clean, organized structure
- Separated concerns (data, code, config, scripts)
- Proper logging and checkpointing
- Comprehensive documentation
- Modular architecture