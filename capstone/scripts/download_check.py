from data.adapters import MathVistaAdapter, EgoSchemaAdapter, VSRAdapter

for AdapterCls, name in [
    (MathVistaAdapter, "MathVista"),
    (EgoSchemaAdapter, "EgoSchema"),
    (VSRAdapter,       "VSR"),
]:
    print(f"\n--- {name} ---")
    count = 0
    for sample in AdapterCls(split="train", max_samples=3):
        print(f"  id={sample['id']}  answer={sample['answer']}  has_image={sample['image'] is not None}")
        count += 1
    print(f"  OK: {count} samples")
