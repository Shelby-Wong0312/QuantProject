"""
Example usage of Capital.com API client
"""
import os
from datetime import datetime, timedelta
from api_client import CapitalComClient, Resolution
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def example_basic_usage():
    """Basic usage example"""
    # Create client instance
    client = CapitalComClient()
    
    # Connect to API
    if client.connect():
        print("Successfully connected to Capital.com API")
        
        # Get accounts
        accounts = client.get_accounts()
        if accounts:
            print(f"\nFound {len(accounts)} accounts:")
            for account in accounts:
                print(f"  - {account.get('accountName')}: {account.get('balance')}")
        
        # Search for Apple stock
        markets = client.search_markets("AAPL")
        if markets:
            print(f"\nFound {len(markets)} markets for 'AAPL':")
            for market in markets[:5]:  # Show first 5
                print(f"  - {market.get('instrumentName')} ({market.get('epic')})")
        
        # Disconnect
        client.disconnect()
        print("\nDisconnected from API")
    else:
        print("Failed to connect to API")


def example_historical_data():
    """Example of fetching historical data"""
    with CapitalComClient() as client:
        # Search for a market
        markets = client.search_markets("EUR/USD")
        if markets and len(markets) > 0:
            epic = markets[0].get('epic')
            print(f"Fetching historical data for {markets[0].get('instrumentName')} ({epic})")
            
            # Get historical prices for last 7 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            prices = client.get_historical_prices(
                epic=epic,
                resolution=Resolution.HOUR,
                from_date=start_date,
                to_date=end_date
            )
            
            if prices:
                price_data = prices.get('prices', [])
                print(f"\nReceived {len(price_data)} price bars")
                
                # Show first 5 bars
                for i, bar in enumerate(price_data[:5]):
                    print(f"\nBar {i+1}:")
                    print(f"  Time: {bar.get('snapshotTime')}")
                    print(f"  Open: {bar.get('openPrice', {}).get('ask')}")
                    print(f"  High: {bar.get('highPrice', {}).get('ask')}")
                    print(f"  Low: {bar.get('lowPrice', {}).get('ask')}")
                    print(f"  Close: {bar.get('closePrice', {}).get('ask')}")


def example_with_context_manager():
    """Example using context manager"""
    # Using context manager for automatic connection/disconnection
    with CapitalComClient() as client:
        # Get open positions
        positions = client.get_positions()
        if positions:
            print(f"\nOpen positions: {len(positions)}")
            for position in positions:
                print(f"  - {position.get('market', {}).get('instrumentName')}: "
                      f"{position.get('direction')} {position.get('size')} @ "
                      f"{position.get('level')}")
        else:
            print("\nNo open positions")


if __name__ == "__main__":
    # Note: Make sure to set environment variables before running
    # CAPITAL_COM_API_KEY=your_api_key
    # CAPITAL_COM_IDENTIFIER=your_identifier
    # CAPITAL_COM_PASSWORD=your_password
    # CAPITAL_COM_DEMO_MODE=True
    
    print("Capital.com API Client Examples")
    print("=" * 50)
    
    # Check if credentials are set
    if not os.getenv("CAPITAL_COM_API_KEY"):
        print("\nError: Please set the following environment variables:")
        print("  - CAPITAL_COM_API_KEY")
        print("  - CAPITAL_COM_IDENTIFIER")
        print("  - CAPITAL_COM_PASSWORD")
        print("  - CAPITAL_COM_DEMO_MODE (optional, defaults to True)")
    else:
        # Run examples
        print("\n1. Basic Usage Example:")
        print("-" * 30)
        example_basic_usage()
        
        print("\n\n2. Historical Data Example:")
        print("-" * 30)
        example_historical_data()
        
        print("\n\n3. Context Manager Example:")
        print("-" * 30)
        example_with_context_manager()