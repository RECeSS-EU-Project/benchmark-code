# Benchmark code for paper "Large-scale benchmark of collaborative filtering applications for drug repurposing: enhancing the evaluation of model performance"

## Install

Please refer to the [README](https://github.com/recess-eu-project/benchscofi) to install cross-platform algorithms.

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

## Benchmark

Execute the following commands

```bash
SAVE_FOLDER="../benchmark-results/" ## or wherever you want

ALG="ALSWR,FastaiCollabWrapper,HAN,LibMF,LogisticMF,NIMCGCN,PMF,SimpleBinaryClassifier,VariationalWrapper,DRRS"
python3 -m main --models "$ALG" --datasets "Gottlieb" --njobs 15 --N 100 --K 5 --splitting "random_simple" --save_folder "$SAVE_FOLDER/results_Gottlieb/"

ALG="ALSWR,FastaiCollabWrapper,HAN,LibMF,LogisticMF,NIMCGCN,PMF"
python3 -m main --models "$ALG" --datasets "Cdataset" --njobs 15 --N 100 --K 5 --splitting "random_simple" --save_folder "$SAVE_FOLDER/results_Cdataset/"

ALG="ALSWR,FastaiCollabWrapper,HAN,LibMF,LogisticMF,NIMCGCN,PMF"
python3 -m main --models "$ALG" --datasets "TRANSCRIPT" --njobs 15 --N 100 --K 5 --splitting "random_simple" --save_folder "$SAVE_FOLDER/results_TRANSCRIPT/"

## after importing the full PREDICT dataset
ALG="ALSWR,FastaiCollabWrapper,HAN,LibMF,LogisticMF,NIMCGCN,PMF"
python3 -m main --models "$ALG" --datasets "PREDICT" --njobs 15 --N 100 --K 5 --splitting "random_simple" --save_folder "$SAVE_FOLDER/results_PREDICT/"

ALG="ALSWR,FastaiCollabWrapper,HAN,LibMF,LogisticMF,NIMCGCN,PMF"
python3 -m main --models "$ALG" --datasets "PREDICT" --njobs 15 --N 100 --K 5 --splitting "random_simple" --save_folder "$SAVE_FOLDER/results_PREDICTpublic/"

ALG="ALSWR,FastaiCollabWrapper,HAN,LibMF,LogisticMF,NIMCGCN,PMF"
python3 -m main --models "$ALG" --datasets "LRSSL" --njobs 15 --N 100 --K 5 --splitting "random_simple" --save_folder "$SAVE_FOLDER/results_LRSSL/"
```