```bash
conda create --name benchmark_code python=3.8 -y
conda activate benchmark_code
python3 -m pip install -r requirements.txt
python3 -m benchmark_pipeline ## test
python3 -m main --models "PMF" --datasets "Synthetic" --N 3 --K 5 --njobs 3 ## run pipeline as described in the paper
```