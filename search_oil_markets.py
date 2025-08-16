"""
Search for Oil Markets on Capital.com
"""

import os
import sys
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set API credentials
# os.environ['CAPITAL_API_KEY'] removed - use .env file
# os.environ['CAPITAL_IDENTIFIER'] removed - use .env file
# os.environ['CAPITAL_API_PASSWORD'] removed - use .env file
os.environ['CAPITAL_DEMO_MODE'] = 'True'

from src.connectors.capital_com_api import CapitalComAPI

def search_oil_markets():
    """Search for all oil-related markets"""
    print("\n" + "="*60)
    print("SEARCHING FOR OIL MARKETS ON CAPITAL.COM")
    print("="*60)
    
    # Initialize and authenticate
    api = CapitalComAPI()
    if not api.authenticate():
        print("[ERROR] Failed to authenticate")
        return
    
    print("[OK] Authenticated successfully\n")
    
    # Search for different oil-related keywords
    search_terms = ['oil', 'crude', 'wti', 'brent', 'commodity', 'energy']
    all_oil_markets = {}
    
    for term in search_terms:
        print(f"\nSearching for '{term}'...")
        result = api.search_markets(term)
        
        if result and 'markets' in result:
            markets = result['markets']
            print(f"  Found {len(markets)} markets")
            
            for market in markets:
                epic = market.get('epic', '')
                name = market.get('instrumentName', '')
                
                # Filter for oil-related markets
                if any(oil_term in name.upper() for oil_term in ['OIL', 'CRUDE', 'WTI', 'BRENT', 'PETROLEUM']):
                    if epic not in all_oil_markets:
                        all_oil_markets[epic] = {
                            'name': name,
                            'epic': epic,
                            'type': market.get('instrumentType', ''),
                            'expiry': market.get('expiry', '-')
                        }
    
    # Display all found oil markets
    print("\n" + "="*60)
    print("ALL OIL-RELATED MARKETS FOUND:")
    print("="*60)
    
    if all_oil_markets:
        for epic, info in all_oil_markets.items():
            print(f"\nEpic: {epic}")
            print(f"  Name: {info['name']}")
            print(f"  Type: {info['type']}")
            print(f"  Expiry: {info['expiry']}")
    else:
        print("\n[WARNING] No oil markets found")
        
    # Try to get specific market details for common oil epics
    print("\n" + "="*60)
    print("CHECKING COMMON OIL EPICS:")
    print("="*60)
    
    common_epics = [
        'OIL_CRUDE',
        'OIL',
        'CC.D.CL.UNC.IP',  # Common WTI futures epic format
        'CC.D.LCO.UNC.IP', # Common Brent futures epic format
    ]
    
    for epic in common_epics:
        print(f"\nChecking {epic}...")
        details = api.get_market_details(epic)
        if details:
            print(f"  [FOUND] Market exists!")
            if 'instrument' in details:
                inst = details['instrument']
                print(f"  Name: {inst.get('name', 'N/A')}")
                print(f"  Type: {inst.get('type', 'N/A')}")
            if 'snapshot' in details:
                snapshot = details['snapshot']
                print(f"  Bid: {snapshot.get('bid', 'N/A')}")
                print(f"  Offer: {snapshot.get('offer', 'N/A')}")
        else:
            print(f"  [NOT FOUND]")
    
    return all_oil_markets

if __name__ == "__main__":
    markets = search_oil_markets()
    
    if markets:
        print("\n" + "="*60)
        print(f"TOTAL OIL MARKETS FOUND: {len(markets)}")
        print("="*60)
        
        # Pick the first WTI market if available
        wti_epic = None
        for epic, info in markets.items():
            if 'WTI' in info['name'].upper() or 'CRUDE' in info['name'].upper():
                wti_epic = epic
                print(f"\n[RECOMMENDED] Use epic: {epic}")
                print(f"  Name: {info['name']}")
                break