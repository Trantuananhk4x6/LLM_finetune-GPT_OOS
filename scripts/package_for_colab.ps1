$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$archive = Join-Path (Get-Location) "sales-finetune-colab-$stamp.zip"
Compress-Archive -Path README.md,requirements-colab.txt,pyproject.toml,configs,src,scripts,data\raw -DestinationPath $archive -CompressionLevel Optimal
Write-Output $archive
