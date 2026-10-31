$ErrorActionPreference='Stop'
$taskOutput=Split-Path -Parent $PSScriptRoot
$taskProject=Split-Path -Parent $taskOutput
$taskSpatial=Join-Path $taskProject '06_研究资料/LARC_NAC_CNP0004138_Phase2_StereoSeq_SpatialLocalization/03_stereoseq_matrix'
Add-Type -Path (Join-Path $PSScriptRoot '04b_spatial_target_rows.cs')
$taskLog=New-Object System.IO.StringWriter
$taskConsole=[Console]::Out
try {
 [Console]::SetOut($taskLog)
 foreach($section in @('CR1','CR2','NR1','NR2')) {
  [SpatialTargetRows]::Extract((Join-Path $taskSpatial ('ST_'+$section+'_expression_clean.tsv')),(Join-Path $taskOutput ('source_data/ST_'+$section+'_LR_gene_rows.tsv')))
 }
} finally {
 [Console]::SetOut($taskConsole)
 $taskLog.ToString() | Set-Content -LiteralPath (Join-Path $taskOutput 'logs/04b_extract_spatial.log') -Encoding utf8
 $taskLog.Dispose()
}
