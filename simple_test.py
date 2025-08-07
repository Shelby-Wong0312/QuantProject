import zmq
import time

print("Simple ZMQ Test - Checking MT4 Connection")
print("="*40)

context = zmq.Context()

# Send a simple subscribe command
print("\n1. Sending SUBSCRIBE command for BTCUSD...")
push = context.socket(zmq.PUSH)
push.connect("tcp://localhost:32768")
push.send_string("SUBSCRIBE_MARKETDATA;BTCUSD")
push.close()

time.sleep(2)

# Try to receive market data
print("\n2. Checking for market data...")
sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:32770")
sub.setsockopt_string(zmq.SUBSCRIBE, "")
sub.setsockopt(zmq.RCVTIMEO, 3000)

try:
    for i in range(5):
        msg = sub.recv_string()
        print(f"Received: {msg[:100]}...")
        if "BTC" in msg:
            print("BTC data found!")
            break
except zmq.Again:
    print("No data received - timeout")
except Exception as e:
    print(f"Error: {e}")

sub.close()

# Send subscribe for common symbols
print("\n3. Trying other symbols...")
push = context.socket(zmq.PUSH)
push.connect("tcp://localhost:32768")

symbols = ['EURUSD', 'GBPUSD', 'GOLD', 'XAUUSD', 'US30', 'SPX500']
for sym in symbols:
    print(f"   Subscribing to {sym}")
    push.send_string(f"SUBSCRIBE_MARKETDATA;{sym}")
    time.sleep(0.5)

push.close()

time.sleep(2)

# Check what we got
print("\n4. Checking subscribed symbols...")
sub = context.socket(zmq.SUB)
sub.connect("tcp://localhost:32770")
sub.setsockopt_string(zmq.SUBSCRIBE, "")
sub.setsockopt(zmq.RCVTIMEO, 2000)

received_symbols = set()
try:
    for i in range(10):
        msg = sub.recv_string()
        # Extract symbol from message
        if ":|:" in msg:
            symbol = msg.split(":|:")[0]
            received_symbols.add(symbol)
except zmq.Again:
    pass

if received_symbols:
    print(f"Active symbols: {', '.join(received_symbols)}")
else:
    print("No active symbols found")

sub.close()
context.term()

print("\n" + "="*40)
print("Test complete")