from pathlib import Path
import sys,shutil,csv
r=Path(__file__).resolve().parents[1];target=Path(sys.argv[1]).resolve();target.mkdir(parents=True,exist_ok=True)
for p in (r/'code').rglob('*'):
 if p.is_file():q=target/p.relative_to(r/'code');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
with (r/'metadata/code_and_source_mapping.tsv').open(encoding='utf8') as f:
 for row in csv.DictReader(f,delimiter='\t'):
  if row['file'].startswith('results/') or row['file'].startswith('frozen_programs/'):
   src=r/row['file'];dest=target/row['source'];dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
print('Available code/aggregate results copied; obtain original source inputs separately and configure legacy constants.')
