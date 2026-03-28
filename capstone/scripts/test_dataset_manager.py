from data.pipeline.dataset_manager import DatasetManager

# Use absolute path to config (works when exec'd from any location)
config_path = r"C:\Users\PESU-RF\Desktop\TEAM58_Capstone\capstone\config\training_config.yaml"

# Use small max_samples during dev — avoids downloading full datasets
manager = DatasetManager.from_config(config_path)

counts = {}
for sample in manager.iter_samples():
    counts[sample["source"]] = counts.get(sample["source"], 0) + 1

print(counts)  # e.g. {'mathvista': 6141, 'egoschema': 5031, 'vsr': 2880}
