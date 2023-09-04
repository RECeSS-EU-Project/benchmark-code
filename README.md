```bash
conda create --name benchmark_code python=3.8 -y
conda activate benchmark_code
python3 -m pip install -r requirements.txt
## test
python3 -m benchmark_pipeline 
## run pipeline as described in the paper 
## for M="PMF", D="Synthetic" N=3 K=5 on 3 parallel jobs
python3 -m main --models "PMF" --datasets "Synthetic" \ 
	--N 3 --K 5 --njobs 3 
```

Benchmark

```bash
ALG="PMF,ALSWR,FastaiCollabWrapper,NIMCGCN,DRRS,SCPMF,BNNR,MBiRW,LogisticMF,DDA_SKF,HAN"
python3 -m main --models "$ALG" --datasets Gottlieb \ 
	--njobs 10 --N 100 --splitting random_simple
```