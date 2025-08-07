import zmq
import time

print("Testing ZeroMQ connection to MT4...")
print("="*50)

# Test if MT4 is listening on port 5555
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.setsockopt(zmq.LINGER, 0)  # Don't wait on close
socket.setsockopt(zmq.RCVTIMEO, 2000)  # 2 second timeout

try:
    print("Connecting to tcp://localhost:5555...")
    socket.connect("tcp://localhost:5555")
    
    # Send a simple message
    msg = '{"command":"HEARTBEAT"}'
    print(f"Sending: {msg}")
    socket.send_string(msg)
    
    # Try to receive
    print("Waiting for response (2 seconds)...")
    reply = socket.recv_string()
    print(f"Received: {reply}")
    print("\n[SUCCESS] MT4 is responding!")
    
except zmq.Again:
    print("\n[TIMEOUT] No response from MT4")
    print("\nPlease check:")
    print("1. Is MT4 running?")
    print("2. Is PythonBridge EA loaded on a chart?")
    print("3. Does EA show a smiley face?")
    print("4. Is AutoTrading enabled (green button)?")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    
finally:
    socket.close()
    context.term()
    print("\nTest completed")