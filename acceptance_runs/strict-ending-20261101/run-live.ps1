$ErrorActionPreference = 'Continue'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
python -u experiments\run_acceptance.py --genre thriller --scenes 8 --premise "Mira investigates a silent archive where Voss has sealed the records and Nia knows how the old index was altered." --output-dir acceptance_runs\strict-ending-20261101 *> acceptance_runs\strict-ending-20261101\live-run.log
Set-Content -Path acceptance_runs\strict-ending-20261101\exit-code.txt -Value $LASTEXITCODE
