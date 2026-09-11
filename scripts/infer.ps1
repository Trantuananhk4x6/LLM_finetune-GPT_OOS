$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src;$env:PYTHONPATH"
$PYTHON_EXEC = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
& $PYTHON_EXEC -m llm_finetune_gpt_oos.main infer --config configs/inference.yaml --prompt $args[0]
