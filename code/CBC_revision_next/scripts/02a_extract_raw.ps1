$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$projectRoot=Split-Path -Parent $taskRoot
$taskLookup=Join-Path $taskRoot 'source_data/raw_gene_lookup.txt'
if(-not (Test-Path -LiteralPath $taskLookup)){throw 'Frozen raw_gene_lookup.txt is required (47 target/marker genes; source gene-row order).'}
Add-Type -Path (Join-Path $taskRoot 'scripts/02a_fast_raw_reader.cs')
$taskLog=[System.IO.StreamWriter]::new((Join-Path $taskRoot 'logs/02a_fast_raw_reader_console.log'),$false)
$taskOldConsole=[Console]::Out
try {
 [Console]::SetOut($taskLog)
 [TargetRawReader]::Run((Join-Path $projectRoot '06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery/02_scRNA_matrix/source/MM_coordinate_matrix.mtx.gz'),$taskLookup,(Join-Path $taskRoot 'analysis/selected_counts.bin'),(Join-Path $taskRoot 'analysis/all_cell_totals.bin'),(Join-Path $taskRoot 'analysis/all_cell_nnz.bin'),47,378344)
} finally {$taskLog.Flush();[Console]::SetOut($taskOldConsole);$taskLog.Dispose()}
