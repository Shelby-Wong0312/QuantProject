"""
Data Validation and Quality Check Module
Phase 1: Ensure data integrity before analysis
"""

import pandas as pd
import numpy as np
import sqlite3
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for quantitative analysis"""
    
    def __init__(self, db_path: str = None):
        """Initialize data validator"""
        if db_path is None:
            # Use absolute path relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(project_root, 'data', 'quant_trading.db')
        else:
            self.db_path = db_path
        self.parquet_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                        'scripts', 'download', 'historical_data', 'daily')
        self.validation_report = {
            'summary': {},
            'issues': [],
            'statistics': {},
            'recommendations': []
        }
        
    def validate_all_stocks(self) -> Dict:
        """Run complete validation on all stocks"""
        logger.info("Starting comprehensive data validation...")
        
        # Get list of all stocks
        stocks = self._get_all_stocks()
        logger.info(f"Found {len(stocks)} stocks to validate")
        
        # Validation results
        results = {
            'total_stocks': len(stocks),
            'valid_stocks': 0,
            'invalid_stocks': 0,
            'missing_data': [],
            'anomalies': [],
            'quality_scores': {}
        }
        
        # Validate each stock
        for symbol in tqdm(stocks, desc="Validating stocks"):
            score, issues = self._validate_stock(symbol)
            results['quality_scores'][symbol] = score
            
            if score >= 0.95:
                results['valid_stocks'] += 1
            else:
                results['invalid_stocks'] += 1
                if issues:
                    results['anomalies'].append({
                        'symbol': symbol,
                        'score': score,
                        'issues': issues
                    })
        
        self.validation_report['summary'] = results
        return results
    
    def _get_all_stocks(self) -> List[str]:
        """Get list of all stocks from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT symbol FROM daily_data ORDER BY symbol")
        stocks = [row[0] for row in cursor.fetchall()]
        conn.close()
        return stocks
    
    def _validate_stock(self, symbol: str) -> Tuple[float, List[str]]:
        """Validate individual stock data"""
        issues = []
        score = 1.0
        
        try:
            # Load data from database
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT date, open_price, high_price, low_price, close_price, volume
                FROM daily_data 
                WHERE symbol = '{symbol}'
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if df.empty:
                issues.append("No data found")
                return 0.0, issues
            
            # Check 1: Data completeness
            expected_days = self._calculate_trading_days()
            actual_days = len(df)
            completeness = actual_days / expected_days
            if completeness < 0.9:
                issues.append(f"Missing data: {completeness:.1%} complete")
                score *= completeness
            
            # Check 2: Price consistency
            price_issues = self._check_price_consistency(df)
            if price_issues:
                issues.extend(price_issues)
                score *= 0.9
            
            # Check 3: Volume validation
            if (df['volume'] == 0).sum() > len(df) * 0.1:
                issues.append("Excessive zero volume days")
                score *= 0.95
            
            # Check 4: Outlier detection
            outliers = self._detect_outliers(df)
            if outliers:
                issues.append(f"Found {len(outliers)} outliers")
                score *= (1 - len(outliers) / len(df))
            
            # Check 5: Data gaps
            gaps = self._check_data_gaps(df)
            if gaps:
                issues.append(f"Found {len(gaps)} data gaps")
                score *= 0.95
                
        except Exception as e:
            issues.append(f"Validation error: {str(e)}")
            score = 0.0
            
        return max(0, min(1, score)), issues
    
    def _calculate_trading_days(self) -> int:
        """Calculate expected number of trading days in 15 years"""
        # Approximately 252 trading days per year
        return 252 * 15
    
    def _check_price_consistency(self, df: pd.DataFrame) -> List[str]:
        """Check OHLC price relationships"""
        issues = []
        
        # High should be >= Low
        invalid_hl = df[df['high_price'] < df['low_price']]
        if not invalid_hl.empty:
            issues.append(f"High < Low in {len(invalid_hl)} records")
        
        # Close should be between High and Low
        invalid_close = df[(df['close_price'] > df['high_price']) | 
                          (df['close_price'] < df['low_price'])]
        if not invalid_close.empty:
            issues.append(f"Close outside H-L range in {len(invalid_close)} records")
        
        # Open should be between High and Low
        invalid_open = df[(df['open_price'] > df['high_price']) | 
                         (df['open_price'] < df['low_price'])]
        if not invalid_open.empty:
            issues.append(f"Open outside H-L range in {len(invalid_open)} records")
        
        # Check for negative prices
        if (df[['open_price', 'high_price', 'low_price', 'close_price']] < 0).any().any():
            issues.append("Negative prices found")
        
        return issues
    
    def _detect_outliers(self, df: pd.DataFrame) -> List[int]:
        """Detect price outliers using statistical methods"""
        outliers = []
        
        # Calculate returns
        df['returns'] = df['close_price'].pct_change()
        
        # Use 5 standard deviations as threshold
        mean_return = df['returns'].mean()
        std_return = df['returns'].std()
        threshold = 5 * std_return
        
        # Find outliers
        outlier_mask = np.abs(df['returns'] - mean_return) > threshold
        outliers = df[outlier_mask].index.tolist()
        
        return outliers
    
    def _check_data_gaps(self, df: pd.DataFrame) -> List[str]:
        """Check for gaps in data timeline"""
        gaps = []
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Check for gaps larger than 10 trading days
        date_diff = df['date'].diff()
        large_gaps = df[date_diff > timedelta(days=14)]  # Accounting for weekends
        
        for idx, row in large_gaps.iterrows():
            gap_size = date_diff.iloc[idx].days
            gaps.append(f"Gap of {gap_size} days at {row['date']}")
        
        return gaps
    
    def generate_quality_report(self) -> str:
        """Generate comprehensive data quality report"""
        logger.info("Generating data quality report...")
        
        # Run validation if not already done
        if not self.validation_report['summary']:
            self.validate_all_stocks()
        
        # Create report
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY VALIDATION REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary statistics
        summary = self.validation_report['summary']
        report.append("SUMMARY")
        report.append("-" * 30)
        report.append(f"Total Stocks: {summary['total_stocks']}")
        report.append(f"Valid Stocks (>95% quality): {summary['valid_stocks']}")
        report.append(f"Invalid Stocks: {summary['invalid_stocks']}")
        report.append(f"Average Quality Score: {np.mean(list(summary['quality_scores'].values())):.2%}")
        report.append("")
        
        # Top issues
        if summary['anomalies']:
            report.append("TOP ISSUES")
            report.append("-" * 30)
            # Sort by score (worst first)
            worst_stocks = sorted(summary['anomalies'], key=lambda x: x['score'])[:10]
            for stock in worst_stocks:
                report.append(f"{stock['symbol']}: Score {stock['score']:.2%}")
                for issue in stock['issues'][:2]:  # Show top 2 issues
                    report.append(f"  - {issue}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 30)
        
        if summary['invalid_stocks'] > summary['total_stocks'] * 0.05:
            report.append("⚠️  More than 5% of stocks have quality issues")
            report.append("   Consider re-downloading problematic stocks")
        
        if summary['anomalies']:
            outlier_stocks = [s['symbol'] for s in summary['anomalies'] 
                            if 'outliers' in str(s['issues'])]
            if outlier_stocks:
                report.append(f"⚠️  {len(outlier_stocks)} stocks have outliers")
                report.append("   Review and potentially clean outlier data")
        
        report.append("")
        report.append("=" * 60)
        
        # Save report
        report_text = "\n".join(report)
        report_path = 'reports/data_quality_report.txt'
        os.makedirs('reports', exist_ok=True)
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Also save detailed JSON report
        json_path = 'reports/data_quality_detailed.json'
        with open(json_path, 'w') as f:
            json.dump(self.validation_report, f, indent=2, default=str)
        
        logger.info(f"Report saved to {report_path}")
        return report_text

def main():
    """Run data validation"""
    validator = DataValidator()
    
    # Run validation
    results = validator.validate_all_stocks()
    
    # Generate report
    report = validator.generate_quality_report()
    print(report)
    
    # Summary
    print(f"\nValidation complete!")
    print(f"Valid stocks: {results['valid_stocks']}/{results['total_stocks']}")
    print(f"Reports saved to 'reports/' directory")

if __name__ == "__main__":
    main()