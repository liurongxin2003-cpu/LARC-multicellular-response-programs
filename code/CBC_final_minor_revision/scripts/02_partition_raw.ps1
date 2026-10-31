$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskProject=Split-Path -Parent $taskRoot
$taskSource=Join-Path $taskProject '06_研究资料/LARC_NAC_CNP0004138_Phase1_ResponseState_Discovery/02_scRNA_matrix/source/MM_coordinate_matrix.mtx.gz'
$taskGroups=Import-Csv -LiteralPath (Join-Path $taskRoot 'source_data/selected_library_groups.tsv') -Delimiter "`t"
New-Item -ItemType Directory -Path (Join-Path $taskRoot 'source_data/library_counts') -Force|Out-Null
Add-Type -Path (Join-Path $PSScriptRoot '02_partition_raw.cs')
$taskCapture=New-Object System.IO.StringWriter
$taskConsole=[Console]::Out
try{[Console]::SetOut($taskCapture);[LibraryPartitionReader]::Run($taskSource,(Join-Path $taskRoot 'source_data/library_lookup.tsv'),(Join-Path $taskRoot 'source_data/library_counts'),$taskGroups.Count,378344)}finally{[Console]::SetOut($taskConsole);$taskCapture.ToString()|Set-Content -LiteralPath (Join-Path $taskRoot 'logs/02_partition_raw.log') -Encoding utf8;$taskCapture.Dispose()}
