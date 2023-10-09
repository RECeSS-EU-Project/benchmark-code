# Benchmark code for paper "Large-scale benchmark of collaborative filtering applications for drug repurposing: enhancing the evaluation of model performance"

## Install

Please refer to the [README](https://github.com/recess-eu-project/benchscofi) to install cross-platform algorithms.

```bash
conda create --name benchmark_code python=3.8 -y
conda activate benchmark_code
python3 -m pip install -r requirements.txt

## test that everything is properly installed
python3 -m benchmark_pipeline 

## Run a 5-fold crossvalidation for PMF on dataset Synthetic 3 times with 3 parallel jobs
python3 -m main --models "PMF" --datasets "Synthetic" \ 
	--N 3 --K 5 --njobs 3 
```

## Benchmark

```bash
SAVE_FOLDER="../benchmark-results/" ## folder where files are saved
## algorithms to test
ALGOS="ALSWR,BNNR,DDA-SKF,FastaiCollabWrapper,HAN,LibMF,LogisticMF,NIMCGCN,PMF" 
DATAS=("Cdataset" "Gottlieb" "LRSSL" "PREDICT" "PREDICT_Gottlieb" "Synthetic" "TRANSCRIPT")

## "Gottlieb" -> "Fdataset"
## "PREDICT_Gottlieb" -> "Gottlieb"

SPLITS=("random_simple" "weakly_correlated")
NJOBS=10 ## number of parallel jobs
N=100 ## number of runs
K=5 ## number of folds in cross-validation

for SPLIT in "${SPLITS[@]}"
do
    for DATA in "${DATAS[@]}"
    do
        echo $SPLIT"----"$DATA;
        python3 -m main --models "$ALGOS" --datasets "$DATA" --njobs "$NJOBS" --N "$N" --K "$K" \
        	--splitting "$SPLIT" --save_folder "$SAVE_FOLDER/results_"$DATA"/";
    done
done