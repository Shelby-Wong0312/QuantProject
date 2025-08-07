import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import requests
    print("OK - requests imported successfully")
except ImportError as e:
    print(f"ERROR - Failed to import requests: {e}")

try:
    import dotenv
    print("OK - dotenv imported successfully")
except ImportError as e:
    print(f"ERROR - Failed to import dotenv: {e}")

try:
    from dotenv import load_dotenv
    print("OK - load_dotenv imported successfully")
except ImportError as e:
    print(f"ERROR - Failed to import load_dotenv: {e}")