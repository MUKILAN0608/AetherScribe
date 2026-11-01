$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -u "C:\Users\Mukil\quantum_rl\experiments\run_acceptance.py" --genre thriller --scenes 8 --premise "Mira investigates a silent archive where Voss has sealed the records and Nia knows how the old index was altered." --output-dir "C:\Users\Mukil\quantum_rl\acceptance_runs\deep-audit-live-20261101" *> "C:\Users\Mukil\quantum_rl\acceptance_runs\deep-audit-live-20261101\live-run.log"
Set-Content -LiteralPath "C:\Users\Mukil\quantum_rl\acceptance_runs\deep-audit-live-20261101\exit-code.txt" -Value $LASTEXITCODE
