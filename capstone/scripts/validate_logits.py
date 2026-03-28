"""
Validation script to check logits output from Phase 1
Run this after Phase 1 completes to verify cache integrity
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
import sys


def validate_logits(logits_dir: str = "outputs/logits") -> Dict:
    """
    Validate all logit cache files in the outputs directory.
    
    Checks:
    - File structure (JSONL format)
    - Record schema completeness
    - Logit value validity (no NaN/inf, correct shapes)
    - Source distribution
    - ID uniqueness
    - Reasoning mask density
    
    Returns:
        Dict with validation results and statistics
    """
    logits_path = Path(logits_dir)
    
    # Check if directory and files exist
    if not logits_path.exists():
        print(f"[ERROR] Directory does not exist: {logits_dir}")
        return {"status": "error", "message": "Directory not found"}
    
    cache_files = sorted(logits_path.glob("topk_logits_*.jsonl"))
    if not cache_files:
        print(f"[WARNING] No cache files found in {logits_dir}")
        return {"status": "no_data", "files_found": 0}
    
    print(f"Found {len(cache_files)} cache files")
    print("-" * 80)
    
    # Validation metrics
    total_records = 0
    valid_records = 0
    invalid_records = []
    
    sources = Counter()
    sample_ids = set()
    duplicate_ids = []
    
    logit_shapes = Counter()
    reasoning_mask_densities = []
    
    errors = []
    warnings = []
    
    # Process each cache file
    for cache_file in cache_files:
        print(f"\nValidating: {cache_file.name}")
        
        try:
            with open(cache_file, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    total_records += 1
                    
                    try:
                        record = json.loads(line.strip())
                    except json.JSONDecodeError as e:
                        errors.append(f"{cache_file.name}:{line_num} - JSON parse error: {e}")
                        continue
                    
                    # Check required fields (actual cache schema from phase1)
                    required_fields = ["sample_id", "source", "topk_logits"]
                    missing_fields = [f for f in required_fields if f not in record]
                    
                    if missing_fields:
                        errors.append(f"{cache_file.name}:{line_num} - Missing fields: {missing_fields}")
                        continue
                    
                    valid_records += 1
                    
                    # Track source distribution
                    source = record.get("source", "unknown")
                    sources[source] += 1
                    
                    # Check ID uniqueness
                    sample_id = record.get("sample_id")
                    if sample_id in sample_ids:
                        duplicate_ids.append(sample_id)
                    sample_ids.add(sample_id)
                    
                    # Validate logit structure
                    top_k_logits = record.get("topk_logits")
                    if isinstance(top_k_logits, list):
                        top_k_logits_array = np.array(top_k_logits, dtype=np.float32)
                        logit_shapes[str(top_k_logits_array.shape)] += 1
                        
                        # Check for NaN/inf values
                        if np.isnan(top_k_logits_array).any() or np.isinf(top_k_logits_array).any():
                            warnings.append(f"{cache_file.name}:{line_num} - Contains NaN/inf values")
                        
                        # Check value ranges (logits typically in [-10, 10])
                        if (top_k_logits_array > 100).any() or (top_k_logits_array < -100).any():
                            warnings.append(f"{cache_file.name}:{line_num} - Logits out of expected range")
                    else:
                        errors.append(f"{cache_file.name}:{line_num} - Invalid logits format (not list)")
                    
                    # Track reasoning mask density (optional field)
                    reasoning_mask = record.get("reasoning_mask")
                    if reasoning_mask is not None:
                        mask_array = np.array(reasoning_mask, dtype=bool)
                        density = np.mean(mask_array)
                        reasoning_mask_densities.append(density)
        
        except IOError as e:
            errors.append(f"{cache_file.name} - IO error: {e}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\n[STATS] File Statistics:")
    print(f"  Total files: {len(cache_files)}")
    print(f"  Total records: {total_records}")
    print(f"  Valid records: {valid_records}")
    print(f"  Invalid records: {total_records - valid_records}")
    
    if total_records > 0:
        validity_rate = 100 * valid_records / total_records
        print(f"  Validity rate: {validity_rate:.1f}%")
    
    print(f"\n[SOURCE] Source Distribution:")
    for source, count in sources.most_common():
        print(f"  {source}: {count} samples")
    
    print(f"\n[SHAPES] Logit Shape Distribution:")
    for shape, count in logit_shapes.most_common():
        print(f"  {shape}: {count} records")
    
    if duplicate_ids:
        print(f"\n[WARNING] Duplicate IDs found: {len(duplicate_ids)} duplicates")
        print(f"  Unique duplicate IDs: {len(set(duplicate_ids))}")
    else:
        print(f"\n[OK] No duplicate IDs (all {len(sample_ids)} IDs unique)")
    
    if reasoning_mask_densities:
        densities = np.array(reasoning_mask_densities)
        print(f"\n[REASONING] Reasoning Mask Statistics:")
        print(f"  Mean density: {np.mean(densities):.2%}")
        print(f"  Median density: {np.median(densities):.2%}")
        print(f"  Min density: {np.min(densities):.2%}")
        print(f"  Max density: {np.max(densities):.2%}")
        print(f"  Std dev: {np.std(densities):.4f}")
        
        # Check if density is reasonable (typically 10-50%)
        if np.mean(densities) < 0.05:
            warnings.append("[WARNING] Reasoning mask density very low (<5%)")
        elif np.mean(densities) > 0.80:
            warnings.append("[WARNING] Reasoning mask density very high (>80%)")
    
    # Print errors and warnings
    if errors:
        print(f"\n[ERRORS] ({len(errors)}):")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
    
    if warnings:
        print(f"\n[WARNINGS] ({len(warnings)}):")
        for warning in warnings[:10]:  # Show first 10
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more warnings")
    
    if not errors and not warnings:
        print("\n[SUCCESS] All validations passed!")
    
    # Return comprehensive report
    return {
        "status": "success" if valid_records > 0 else "no_data",
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": total_records - valid_records,
        "files_validated": len(cache_files),
        "sources": dict(sources),
        "unique_ids": len(sample_ids),
        "duplicate_ids": len(duplicate_ids),
        "logit_shapes": dict(logit_shapes),
        "reasoning_mask_stats": {
            "count": len(reasoning_mask_densities),
            "mean": float(np.mean(reasoning_mask_densities)) if reasoning_mask_densities else None,
            "median": float(np.median(reasoning_mask_densities)) if reasoning_mask_densities else None,
            "min": float(np.min(reasoning_mask_densities)) if reasoning_mask_densities else None,
            "max": float(np.max(reasoning_mask_densities)) if reasoning_mask_densities else None,
        },
        "errors": errors,
        "warnings": warnings,
    }


if __name__ == "__main__":
    logits_dir = "outputs/logits" if len(sys.argv) < 2 else sys.argv[1]
    result = validate_logits(logits_dir)
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)
