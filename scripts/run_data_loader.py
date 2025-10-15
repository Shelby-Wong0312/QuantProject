"""
Run Historical Data Loader
Cloud DE - Task DE-601
Main script to execute the complete data loading and validation pipeline
"""

import asyncio
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("\n" + "=" * 80)
print("TASK DE-601: HISTORICAL DATA LOADING & VALIDATION")
print("Cloud DE - Loading 15 Years of Real Market Data")
print("=" * 80)
print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)


async def main():
    """Main execution function"""

    try:
        # Step 1: Load Historical Data
        print("\n[STEP 1/3] Loading Historical Data...")
        print("-" * 40)

        from scripts.data_loader.historical_data_loader import HistoricalDataLoader

        loader = HistoricalDataLoader()

        # Run the data loading pipeline
        data_report = await loader.run_full_pipeline()

        print("\nData Loading Complete!")
        print(f"  - Success Rate: {data_report['summary']['success_rate']}")
        print(
            f"  - Average Quality: {data_report['quality_metrics']['average_quality_score']:.2f}"
        )

        # Step 2: Optimize Storage
        print("\n[STEP 2/3] Optimizing Data Storage...")
        print("-" * 40)

        from scripts.data_loader.data_storage import DataStorage

        storage = DataStorage()

        # Optimize database
        storage.optimize_storage()

        # Get storage stats
        stats = storage.get_storage_stats()
        print("Storage Optimization Complete!")
        print(f"  - Database Size: {stats['sqlite_size']:.2f} MB")
        print(f"  - Total Symbols: {stats['total_symbols']}")
        print(f"  - Total Records: {stats['total_records']:,}")

        # Create data catalog
        catalog = storage.create_data_catalog()
        print(f"  - Data Catalog Created: {len(catalog)} symbols")

        # Step 3: Validate Models with Real Data
        print("\n[STEP 3/3] Validating Models with Real Data...")
        print("-" * 40)

        from scripts.validation.model_validation import ModelValidation

        validator = ModelValidation()

        # Get top symbols for validation
        if not catalog.empty:
            # Use top 20 symbols by data completeness
            top_symbols = catalog.nlargest(20, "total_days")["symbol"].tolist()
        else:
            # Fallback symbols
            top_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]

        # Validate portfolio
        validation_report = validator.validate_portfolio(top_symbols[:10])

        print("Model Validation Complete!")
        print(f"  - Pass Rate: {validation_report['summary']['pass_rate']}")
        print(
            f"  - Avg Annual Return: {validation_report['portfolio_metrics']['avg_annual_return']:.2%}"
        )
        print(
            f"  - Avg Sharpe Ratio: {validation_report['portfolio_metrics']['avg_sharpe_ratio']:.2f}"
        )

        # Generate final report
        validator.generate_performance_report()

        # Summary
        print("\n" + "=" * 80)
        print("TASK DE-601 COMPLETION SUMMARY")
        print("=" * 80)

        # Check success criteria
        success_criteria = {
            "Data Completeness > 95%": float(
                data_report["summary"]["success_rate"].rstrip("%")
            )
            > 95,
            "Data Quality > 90%": data_report["quality_metrics"][
                "average_quality_score"
            ]
            > 0.90,
            "Annual Return > 15%": validation_report["portfolio_metrics"][
                "avg_annual_return"
            ]
            > 0.15,
            "Sharpe Ratio > 1.0": validation_report["portfolio_metrics"][
                "avg_sharpe_ratio"
            ]
            > 1.0,
            "Max Drawdown < 15%": validation_report["portfolio_metrics"][
                "avg_max_drawdown"
            ]
            < 0.15,
            "Win Rate > 55%": validation_report["portfolio_metrics"]["avg_win_rate"]
            > 0.55,
        }

        print("\nSuccess Criteria:")
        all_passed = True
        for criterion, passed in success_criteria.items():
            status = "PASS" if passed else "FAIL"
            symbol = "[OK]" if passed else "[FAIL]"
            print(f"  {symbol} {criterion}: {status}")
            if not passed:
                all_passed = False

        print("\n" + "=" * 80)
        print("DELIVERABLES CREATED:")
        print("=" * 80)

        deliverables = [
            (
                "1. Historical Data Loader",
                "scripts/data_loader/historical_data_loader.py",
            ),
            ("2. Data Storage System", "scripts/data_loader/data_storage.py"),
            ("3. Model Validation", "scripts/validation/model_validation.py"),
            ("4. SQLite Database", "data/historical_market_data.db"),
            ("5. Data Catalog", "data/data_catalog.csv"),
            ("6. Validation Report", "data/data_validation_report.json"),
            ("7. Backtest Results", "data/real_backtest_results.json"),
            ("8. Performance Report", "reports/model_validation_report.md"),
        ]

        for name, path in deliverables:
            exists = "[EXISTS]" if os.path.exists(path) else "[PENDING]"
            print(f"  {exists} {name}")
            print(f"     Path: {path}")

        print("\n" + "=" * 80)
        if all_passed:
            print("FINAL STATUS: [PASS] ALL CRITERIA MET - READY FOR PRODUCTION")
        else:
            print("FINAL STATUS: [WARNING] SOME CRITERIA NOT MET - OPTIMIZATION NEEDED")
        print("=" * 80)
        print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        return all_passed

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the async main function
    success = asyncio.run(main())

    # Exit with appropriate code
    sys.exit(0 if success else 1)
