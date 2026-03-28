import os
import sys

# 1. Get the path to the 'capstone' subdirectory
# Since the script is in /scripts, and 'capstone' is a sibling to /scripts:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CAPSTONE_DIR = os.path.join(PROJECT_ROOT, 'capstone')

# 2. Add 'capstone' to the path so 'from data.adapters' works
if CAPSTONE_DIR not in sys.path:
    sys.path.insert(0, CAPSTONE_DIR)

print(f"Debug: Now looking inside {CAPSTONE_DIR}")

# 3. Try the import again
try:
    from data.adapters import MathVistaAdapter, EgoSchemaAdapter, VSRAdapter
    print("✅ Success! Adapters imported.")
except ModuleNotFoundError as e:
    print(f"❌ Still failing. Content of capstone dir: {os.listdir(CAPSTONE_DIR)}")
    sys.exit(1)

# --- RUN THE TEST ---
for AdapterCls, name in [(MathVistaAdapter, "MathVista"), (EgoSchemaAdapter, "EgoSchema"), (VSRAdapter, "VSR")]:
    print(f"\n--- {name} ---")
    try:
        # Checking if data is actually there
        for i, sample in enumerate(AdapterCls(split="train", max_samples=1)):
            print(f"  Success! Found sample ID: {sample['id']}")
            print(f"  Image object: {type(sample['image'])}")
    except Exception as e:
        print(f"  Error loading {name}: {e}")