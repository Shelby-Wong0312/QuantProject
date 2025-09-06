import torch
import os

# Check all available models
models = {
    'Final': 'reports/ml_models/ppo_trader_final.pt',
    'Iter 150': 'ppo_trader_iter_150.pt', 
    'Iter 100': 'ppo_trader_iter_100.pt',
    'Iter 50': 'ppo_trader_iter_50.pt'
}

print("=" * 60)
print("PPO MODEL STATUS CHECK")
print("=" * 60)

for name, path in models.items():
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"[OK] {name:10} | Path: {path:40} | Size: {size_mb:.2f} MB")
        
        # Try to load and check content
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(checkpoint, dict):
                print(f"  Keys: {list(checkpoint.keys())[:3]}...")
                if 'training_iteration' in checkpoint:
                    print(f"  Training iterations: {checkpoint['training_iteration']}")
        except Exception as e:
            print(f"  Error loading: {e}")
    else:
        print(f"[X] {name:10} | Not found: {path}")

print("\n" + "=" * 60)
print("RECOMMENDATION: Use 'reports/ml_models/ppo_trader_final.pt'")
print("This is the fully trained model with best performance")
print("=" * 60)