import socket
import zmq

print("Checking MT4 port configuration...")
print("="*50)

# Check if ports are in use
ports_to_check = [5555, 5556, 5557, 5558]

print("\n1. Checking if ports are in use:")
for port in ports_to_check:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    if result == 0:
        print(f"   Port {port}: IN USE (something is listening)")
    else:
        print(f"   Port {port}: FREE (nothing listening)")
    sock.close()

print("\n2. MT4 EA Port Configuration:")
print("   The PythonBridge EA should use these ports:")
print("   - REP socket (receive from Python): 5555")
print("   - PUB socket (send to Python): 5557")

print("\n3. Testing alternative connection methods:")

# Try connecting as SUB to see if MT4 is publishing
context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.setsockopt(zmq.RCVTIMEO, 2000)
sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")

try:
    print("\n   Trying SUB connection to port 5557...")
    sub_socket.connect("tcp://localhost:5557")
    data = sub_socket.recv_string()
    print(f"   [SUCCESS] Received data: {data[:50]}...")
except zmq.Again:
    print("   [TIMEOUT] No data from port 5557")
except Exception as e:
    print(f"   [ERROR] {e}")
finally:
    sub_socket.close()

# Try binding instead of connecting (in case EA expects Python to bind)
rep_socket = context.socket(zmq.REP)
rep_socket.setsockopt(zmq.RCVTIMEO, 2000)

try:
    print("\n   Trying to BIND on port 5556 (Python as server)...")
    rep_socket.bind("tcp://*:5556")
    print("   Waiting for MT4 to connect...")
    msg = rep_socket.recv_string()
    print(f"   [SUCCESS] Received: {msg}")
    rep_socket.send_string('{"status":"ok"}')
except zmq.Again:
    print("   [TIMEOUT] MT4 did not connect")
except zmq.error.ZMQError as e:
    if e.errno == 48:  # Address already in use
        print("   [INFO] Port 5556 already in use")
    else:
        print(f"   [ERROR] {e}")
except Exception as e:
    print(f"   [ERROR] {e}")
finally:
    rep_socket.close()
    context.term()

print("\n" + "="*50)
print("Diagnosis complete")
print("\nIf ports are not in use, MT4 EA may not be running.")
print("If ports are in use but no response, check EA configuration.")