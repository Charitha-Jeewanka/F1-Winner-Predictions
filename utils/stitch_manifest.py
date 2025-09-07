from pathlib import Path
import pandas as pd

save_root = Path("./data/raw")

manifest_files = list(save_root.rglob("*/manifest.csv"))

rows = []
for mf in manifest_files:
    try:
        df = pd.read_csv(mf)
        rows.append(df)
    except Exception as e:
        print(f"Skipping {mf}: {e}")

if rows:
    out_df = pd.concat(rows, ignore_index=True)
    out_dir = save_root / "_manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "global_manifest.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Global manifest created: {out_path}")
else:
    print("No manifests found.")
