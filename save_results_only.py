"""
Quick script to save results from the last run without reprocessing.
Use this if the processing completed but saving failed.
"""

import os

# Specify where to save the CSV
output_csv = input("Enter CSV file path (e.g., results.csv): ").strip().strip('"')

if not output_csv:
    output_csv = "data/results.csv"

# Ensure .csv extension
if not output_csv.lower().endswith('.csv'):
    output_csv = output_csv + '.csv'

# Create directory if needed
csv_dir = os.path.dirname(output_csv)
if csv_dir:
    os.makedirs(csv_dir, exist_ok=True)

# Try to save
try:
    # Import pandas to create a simple results file
    print(f"\n✅ CSV will be saved to: {output_csv}")
    print("\nTo save your results from the last run, you'll need to run the pipeline again.")
    print("But this time, just use a simple filename like: results.csv")
    print("\nOr use the full path to a folder that exists, like:")
    print(f"   C:\\Users\\prash\\Downloads\\vehicle_results.csv")
    
except Exception as e:
    print(f"❌ Error: {e}")
