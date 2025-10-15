import json

with open("scripts/download/download_checkpoint.json") as f:
    json.load(f)

completed = len(data["completed"])
failed = len(data["failed"])

print("=" * 50)
print("DOWNLOAD STATUS REPORT")
print("=" * 50)
print(f"Completed: {completed}/4215 ({completed/4215*100:.1f}%)")
print(f"Failed: {failed}")

if completed >= 4215:
    print("\n✅ DOWNLOAD COMPLETE!")
else:
    print(f"\n⏳ Still need to download {4215-completed} more stocks")
