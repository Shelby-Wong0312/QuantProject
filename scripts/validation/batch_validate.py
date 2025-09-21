#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch validation for all stocks - MAIN PRIORITY
"""

import os
import sys
import json
import time
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from quantproject.capital_service import CapitalService

def main():
    """Main validation function - MUST COMPLETE ALL 7000+ STOCKS"""
    
    print("="*60)
    print("STARTING BATCH VALIDATION OF ALL STOCKS")
    print("="*60)
    
    # Initialize service
    service = CapitalService()
    
    # Login
    success, msg = service.login()
    if not success:
        print(f"Login failed: {msg}")
        return
    print("[OK] Logged in successfully")
    
    # Load all tickers
    all_tickers = []
    
    # Load from invalid_tickers.txt
    if os.path.exists('invalid_tickers.txt'):
        with open('invalid_tickers.txt', 'r') as f:
            invalid = [line.strip() for line in f if line.strip()]
            all_tickers.extend(invalid)
            print(f"[OK] Loaded {len(invalid)} tickers from invalid_tickers.txt")
    
    # Load from valid_tickers.txt
    if os.path.exists('valid_tickers.txt'):
        with open('valid_tickers.txt', 'r') as f:
            valid = [line.strip() for line in f if line.strip()]
            all_tickers.extend(valid)
            print(f"[OK] Loaded {len(valid)} tickers from valid_tickers.txt")
    
    # Remove duplicates
    all_tickers = list(set(all_tickers))
    total_count = len(all_tickers)
    print(f"\n[STATS] TOTAL UNIQUE TICKERS TO VALIDATE: {total_count}")
    
    # Load checkpoint
    checkpoint_file = 'batch_validation_checkpoint.json'
    validated = {}
    
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                validated = json.load(f)
            print(f"[OK] Resumed from checkpoint: {len(validated)} already validated")
        except:
            validated = {}
    
    # Get pending tickers
    pending = [t for t in all_tickers if t not in validated]
    print(f"[LIST] Remaining to validate: {len(pending)}")
    
    if not pending:
        print("\n[DONE] ALL TICKERS ALREADY VALIDATED!")
    else:
        print(f"\n[START] Starting validation of {len(pending)} tickers...")
        print("-"*60)
    
    # Statistics
    stats = {
        'start_time': time.time(),
        'valid': 0,
        'tradable': 0,
        'invalid': 0,
        'errors': 0
    }
    
    # Process each ticker
    for idx, ticker in enumerate(pending):
        try:
            # Search for ticker
            search_results = service.search_stocks(ticker, limit=2)
            
            if search_results:
                # Get epic
                epic = None
                for market in search_results:
                    if ticker.upper() in market.get('displayName', '').upper() or \
                       ticker.upper() in market.get('instrumentName', '').upper():
                        epic = market.get('epic')
                        break
                
                if not epic and search_results:
                    epic = search_results[0].get('epic')
                
                if epic:
                    # Get details
                    details = service.get_market_details(epic)
                    if details:
                        is_tradable = details.get('snapshot', {}).get('marketStatus') == 'TRADEABLE'
                        
                        validated[ticker] = {
                            'valid': True,
                            'tradable': is_tradable,
                            'epic': epic,
                            'name': details.get('instrument', {}).get('name', ''),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        stats['valid'] += 1
                        if is_tradable:
                            stats['tradable'] += 1
                        
                        status = "[OK] TRADABLE" if is_tradable else "[OK] Valid"
                    else:
                        validated[ticker] = {'valid': False, 'reason': 'No details'}
                        stats['invalid'] += 1
                        status = "[X] No details"
                else:
                    validated[ticker] = {'valid': False, 'reason': 'No epic'}
                    stats['invalid'] += 1
                    status = "[X] No epic"
            else:
                validated[ticker] = {'valid': False, 'reason': 'Not found'}
                stats['invalid'] += 1
                status = "[X] Not found"
                
        except Exception as e:
            validated[ticker] = {'valid': False, 'reason': str(e)}
            stats['errors'] += 1
            status = f"[X] Error"
        
        # Progress update
        current_total = len(validated)
        progress = current_total / total_count * 100
        
        print(f"[{current_total}/{total_count}] {progress:.1f}% | {ticker}: {status}")
        
        # Save checkpoint every 25 tickers
        if (idx + 1) % 25 == 0:
            with open(checkpoint_file, 'w') as f:
                json.dump(validated, f, indent=2)
            
            # Show statistics
            elapsed = time.time() - stats['start_time']
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(pending) - idx - 1) / rate if rate > 0 else 0
            
            print(f"\n[STATS] Progress Report:")
            print(f"  Validated: {current_total}/{total_count} ({progress:.1f}%)")
            print(f"  Valid: {stats['valid']} | Tradable: {stats['tradable']} | Invalid: {stats['invalid']}")
            print(f"  Rate: {rate:.1f} tickers/sec | ETA: {eta/60:.1f} minutes")
            print("-"*60)
        
        # Rate limiting
        time.sleep(0.15)
    
    # Final save
    with open(checkpoint_file, 'w') as f:
        json.dump(validated, f, indent=2)
    
    # Generate final report
    print("\n" + "="*60)
    print("VALIDATION COMPLETE!")
    print("="*60)
    
    # Calculate final statistics
    final_valid = sum(1 for v in validated.values() if v.get('valid'))
    final_tradable = sum(1 for v in validated.values() if v.get('tradable'))
    tradable_list = [k for k, v in validated.items() if v.get('tradable')]
    
    # Save reports
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'total_tickers': total_count,
        'total_validated': len(validated),
        'valid': final_valid,
        'tradable': final_tradable,
        'invalid': len(validated) - final_valid,
        'valid_rate': f"{final_valid/len(validated)*100:.1f}%",
        'tradable_rate': f"{final_tradable/len(validated)*100:.1f}%",
        'tradable_tickers': tradable_list
    }
    
    with open('FINAL_VALIDATION_REPORT.json', 'w') as f:
        json.dump(final_report, f, indent=2)
    
    with open('TRADABLE_TICKERS.txt', 'w') as f:
        for ticker in tradable_list:
            f.write(f"{ticker}\n")
    
    # Print summary
    print(f"[STATS] FINAL RESULTS:")
    print(f"  Total Validated: {len(validated)}")
    print(f"  Valid: {final_valid} ({final_report['valid_rate']})")
    print(f"  Tradable: {final_tradable} ({final_report['tradable_rate']})")
    print(f"  Invalid: {final_report['invalid']}")
    print(f"\n[FILES] Reports saved:")
    print(f"  - FINAL_VALIDATION_REPORT.json")
    print(f"  - TRADABLE_TICKERS.txt")
    print(f"  - batch_validation_checkpoint.json")
    
    if tradable_list:
        print(f"\n[TOP] Sample tradable tickers:")
        for ticker in tradable_list[:10]:
            info = validated[ticker]
            print(f"  - {ticker}: {info.get('name', 'N/A')}")

if __name__ == "__main__":
    main()