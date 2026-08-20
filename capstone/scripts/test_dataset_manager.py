from data.pipeline.dataset_manager import DatasetManager

# Relative to the project's config path convention (run from the capstone/ directory),
# matching every other pipeline entrypoint (phase1/phase2/phase3).
config_path = "config/training_config.yaml"

# Use small max_samples during dev — avoids downloading full datasets
manager = DatasetManager.from_config(config_path)

counts = {}
for sample in manager.iter_samples():
    counts[sample["source"]] = counts.get(sample["source"], 0) + 1

print(counts)  # e.g. {'mathvista': 6141, 'egoschema': 5031, 'vsr': 2880}
