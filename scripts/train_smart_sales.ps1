$ErrorActionPreference = "Stop"

# Thêm thư mục src vào PYTHONPATH để Python luôn tìm thấy package llm_finetune_gpt_oos
$env:PYTHONPATH = "src;$env:PYTHONPATH"

# Ưu tiên sử dụng Python trong môi trường ảo .venv nếu có
$PYTHON_EXEC = "python"
if (Test-Path ".venv\Scripts\python.exe") {
    $PYTHON_EXEC = ".venv\Scripts\python.exe"
}

Write-Host "[*] Su dung Python tai: $PYTHON_EXEC" -ForegroundColor Cyan
Write-Host "[*] Bat dau chay huan luyen QLoRA ban hang thong minh..." -ForegroundColor Green

& $PYTHON_EXEC -m llm_finetune_gpt_oos.main train --config configs/smart_sales_train.yaml
