"""Quick script to check pipeline results."""
import pandas as pd
import json

# Read assignments
assignments = pd.read_parquet("data/assignments.parquet")
print("=" * 60)
print("ASSIGNMENTS SUMMARY")
print("=" * 60)
print(f"Total segments: {len(assignments)}")
print(f"\nNet distribution:")
print(assignments['net_id'].value_counts())
print(f"\nSubnet distribution:")
print(assignments['subnet_id'].value_counts())

# Check confidence scores
print(f"\nConfidence stats:")
print(assignments['confidence'].describe())

# Read topic hierarchy
with open("data/topic_hierarchy.json") as f:
    hierarchy = json.load(f)

print("\n" + "=" * 60)
print("TOPIC HIERARCHY")
print("=" * 60)
for net_id, net_data in hierarchy["nets"].items():
    print(f"\nNet: {net_data['label']} (ID: {net_id})")
    print(f"  Seeds: {len(net_data['seeds'])} phrases")
    print(f"  Examples: {len(net_data.get('examples', []))} segments")
    print(f"  Subnets: {len(net_data.get('subnets', {}))}")

# Show sample assignments
print("\n" + "=" * 60)
print("SAMPLE ASSIGNMENTS (first 10)")
print("=" * 60)
print(assignments[['segment_id', 'net_id', 'subnet_id', 'confidence']].head(10))

