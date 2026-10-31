from pathlib import Path
import json,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];ROOT=R.parent;P1=ROOT/'06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery';S=P1/'02_scRNA_matrix/source'
genes=json.loads((R/'source_data/raw_target_genes.json').read_text(encoding='utf8'));barcodes=pd.read_csv(S/'MM_coordinate_matrix_barcodes.csv',header=None).iloc[:,0].astype(str).to_numpy();m=pd.read_csv(P1/'02_scRNA_matrix/CNP0004138_scRNA_cell_metadata_clean.tsv',sep='\t');assert np.array_equal(barcodes,m.barcode)
c=np.fromfile(R/'analysis/selected_counts.bin',dtype='<u8').reshape(len(genes),len(m));total=np.fromfile(R/'analysis/all_cell_totals.bin',dtype='<u8');nnz=np.fromfile(R/'analysis/all_cell_nnz.bin',dtype='<u8');assert np.array_equal(total,m.total_counts.to_numpy());assert np.array_equal(nnz,m.n_genes_by_nnz.to_numpy())
np.savez_compressed(R/'analysis/raw_target_gene_counts.npz',counts=c,genes=np.array(genes),barcodes=barcodes.astype(str));m.to_csv(R/'source_data/cell_metadata_linked.tsv',sep='\t',index=False)
names=pd.read_csv(S/'MM_coordinate_matrix_genes.csv',header=None).iloc[:,0].astype(str).str.upper()
audit=dict(source=str(S/'MM_coordinate_matrix.mtx.gz'),raw_dimensions=[55697,378344,445213600],genes_requested=len(genes),gene_source_rows={g:int(names.eq(g).sum()) for g in genes},barcode_order_exact=True,all_cell_total_count_error=0,all_cell_nnz_gene_error=0,parser='C# integer streaming, gzip, all 445213600 entries; Python cross-check and NPZ',source_all_cells=len(m),target_counts_scope='all source cells; downstream restricted to PRET labeled patients')
(R/'qa/raw_extraction_validation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(audit,indent=2))
