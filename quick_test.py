import zmq
import time

print("Quick MT4 Connection Test")
print("="*40)

# Test ports
ports = {
    "PUSH": 32768,
    "PULL": 32769,
    "SUB": 32770
}

context = zmq.Context()

# Test 1: Try to connect to PUSH port
print("\n1. Testing PUSH connection (port 32768)...")
push_socket = context.socket(zmq.PUSH)
push_socket.setsockopt(zmq.SNDTIMEO, 2000)
try:
    push_socket.connect("tcp://localhost:32768")
    print("   [OK] Connected to PUSH port")
    
    # Send a test message
    push_socket.send_string("HEARTBEAT;", zmq.DONTWAIT)
    print("   [OK] Message sent")
except Exception as e:
    print(f"   [ERROR] {e}")
finally:
    push_socket.close()

# Test 2: Try PULL port
print("\n2. Testing PULL connection (port 32769)...")
pull_socket = context.socket(zmq.PULL)
pull_socket.setsockopt(zmq.RCVTIMEO, 2000)
try:
    pull_socket.connect("tcp://localhost:32769")
    print("   [OK] Connected to PULL port")
    
    # Try to receive
    print("   Waiting for data (2 seconds)...")
    msg = pull_socket.recv_string()
    print(f"   [SUCCESS] Received: {msg}")
except zmq.Again:
    print("   [TIMEOUT] No data received")
except Exception as e:
    print(f"   [ERROR] {e}")
finally:
    pull_socket.close()

# Test 3: Try SUB port
print("\n3. Testing SUB connection (port 32770)...")
sub_socket = context.socket(zmq.SUB)
sub_socket.setsockopt(zmq.RCVTIMEO, 2000)
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
try:
    sub_socket.connect("tcp://localhost:32770")
    print("   [OK] Connected to SUB port")
    
    # Try to receive market data
    print("   Waiting for market data (2 seconds)...")
    msg = sub_socket.recv_string()
    print(f"   [SUCCESS] Received: {msg}")
except zmq.Again:
    print("   [TIMEOUT] No market data")
except Exception as e:
    print(f"   [ERROR] {e}")
finally:
    sub_socket.close()

context.term()

print("\n" + "="*40)
print("Diagnosis:")
print("If all connections OK but no data:")
print("-> MT4 EA is not sending data")
print("If connection errors:")
print("-> MT4 EA is not running")