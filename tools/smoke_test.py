from pathlib import Path
import ast,json,csv,hashlib
import pandas as pd
r=Path(__file__).resolve().parents[1]
for p in (r/'code').rglob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
for p in (r/'results').rglob('*.tsv'):
 d=pd.read_csv(p,sep='\t');assert len(d)>0 and d.columns.is_unique
 assert not set(d.columns)&{'patient_id','sample_id','barcode','pseudobulk_id','source_cell_index'}
assert len(pd.read_csv(r/'results/R17/Revision_Supplementary_Table_R17d_LR_sensitivity.tsv',sep='\t'))==10
assert len(pd.read_csv(r/'results/R17/Revision_Supplementary_Table_R17g_fixed_program_sensitivity.tsv',sep='\t'))==12
n=pd.read_csv(r/'results/R1_R12/exact_reselection_null_statistics.tsv',sep='\t');assert (n.groupby('program').size()==462).all()
for row in csv.DictReader((r/'checksums.tsv').open(encoding='utf8'),delimiter='\t'):
 assert hashlib.sha256((r/row['file']).read_bytes()).hexdigest()==row['sha256']
print('Public archive smoke check passed: syntax, table schemas, 462 assignments and file hashes. Not an end-to-end raw-data rerun.')
