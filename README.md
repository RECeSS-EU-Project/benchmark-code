# Benchmark code for "Comprehensive evaluation of collaborative filtering in drug repurposing"

**License**: MIT

**Citation**:

```bash
@article{reda2025comprehensive,
  title={Comprehensive evaluation of pure and hybrid collaborative filtering in drug repurposing},
  author={R{\'e}da, Cl{\'e}mence and Vie, Jill-J{\^e}nn and Wolkenhauer, Olaf},
  journal={Scientific Reports},
  volume={15},
  number={1},
  pages={2711},
  year={2025},
  publisher={Nature Publishing Group UK London}
}

```

## 1. System requirements

**OS**: Linux Debian
**Python version**: 3.8.*

The dependencies, along with package versions, are listed in the file named *requirements.txt*.

## 2. Installation guide (estimated install time: ~30min)

Please refer to the [README](https://github.com/recess-eu-project/benchscofi) to install cross-platform algorithms in package **benchscofi**. It is strongly advised to install the [Conda](https://docs.anaconda.com/free/miniconda/miniconda-install/) tool to create a virtual environment, and [Pip](https://pip.pypa.io/en/stable/installation/) for installing dependencies:

```bash
conda create --name benchmark_code python=3.8 -y
conda activate benchmark_code
```

Once the virtual environment is created:

```bash
python3 -m pip install -r requirements.txt

## Test that everything is properly installed
python3 -m benchmark_pipeline 
```

## 3. Demo (estimated run time: <1min)

### 3.1 Instructions to run on demo data

To run a 5-fold crossvalidation for PMF on dataset Synthetic 3 times with 3 parallel jobs

```bash
python3 -m main --models "PMF" --datasets "Synthetic" --splitting random_simple --N 3 --K 5 --njobs 3 --save_folder "./"
```

### 3.2 Expected output

In the current directory, a folder named "results\_PMF" has appeared, containing a boxplot (.png) that shows the variation of each validation metric across N=3 iterations, a .json file containing the parameters with which the benchmark was run, and N+2 .csv files. N=3 .csv files contain the individual run times and validation metrics for each iteration, whereas the .csv file starting with "seeds\_" contains the random seeds for each iteration and the .csv file starting with "results\_" contains the concatenated values of run times and validation metrics for each iteration.

## 4. Instructions for use

### 4.a Running the software on your own data

On your own implementation of an algorithm (named ALGO, a Python class contains a method "fit" and "predict\_proba" similarly to models in [scikit-learn](https://scikit-learn.org/stable/) with a dictionary of parameters PARAMS and/or your own drug repurposing dataset (with association matrix A, drug feature matrix S and disease feature matrix P as described in the paper), the minimal version of the benchmark for a single iteration with random seed SEED and splitting approach SPLIT (SPLIT="random\_simple" or "weakly\_correlated") is:

```bash
import stanscofi.datasets
import stanscofi.training_testing
import stanscofi.validation
from benchscofi import rowwise_metrics
import numpy as np
import random
from time import time
import gc
import pandas as pd

K=5
ptest=0.2
metric="AUC"

np.random.seed(SEED)
random.seed(SEED)

## Dataset
dataset = stanscofi.datasets.Dataset({"ratings": A, "items": S, "users": P})
split_method = "stanscofi.training_testing."+SPLIT+"_split"
(traintest_folds, val_folds), _ = eval(split_method)(
	dataset, ptest, metric="euclidean", random_state=SEED
)

## Training (cross-validation)
dataset_traintest = dataset.subset(traintest_folds)
start_time = time()
di_results = stanscofi.training_testing.cv_training(
	ALGO, PARAMS, dataset_traintest, K, metric, 
	k=1, beta=1, threshold=0, cv_type="random", 
	random_state=SEED
)
runtime = time()-start_time

## Validation
best_id = np.argmax(di_results["test_metric"])
best_model = di_results["models"][best_id]
dataset_val = dataset.subset(val_folds)
p_start_time = time()
scores = best_model.predict_proba(dataset_val)
p_runtime = time()-p_start_time
best_model.print_scores(scores)
predictions = best_model.predict(scores, threshold=0)
best_model.print_classification(predictions)

## Compute metrics

### 1. row-wise AUC (using scikit-learn) and row-wise NDCG@#items
metrics, plot_args = stanscofi.validation.compute_metrics(
	scores, predictions, dataset_val, 
	metrics=["AUC", "NDCGk", "Fscore"], 
	k=dataset_val.nitems, beta=1, verbose=False
)
di_metrics = metrics.iloc[:-1,:].to_dict()["Average"]
di_metrics.setdefault("training time (sec)", runtime)
di_metrics.setdefault("prediction time (sec)", p_runtime)

### 2. row-wise disagreeing AUC
lin_aucs = rowwise_metrics.calc_auc(
	scores, dataset_val, 
	transpose=False, verbose=False
)
lin_auc = np.mean(lin_aucs) if (np.max(lin_aucs)>0) else 0.5
di_metrics.setdefault("Lin's AUC", lin_auc)

### 3. global AUC and global NDCG@#pairs
y_val = (dataset_val.folds.toarray()*dataset_val.ratings.toarray()).ravel()
y_val[y_val<1] = 0
y_pred = scores.toarray().ravel()
auc = AUC(y_val, y_pred, 1, 1)
ndcg = NDCGk(y_val, y_pred, y_pred.shape[0], 1)

di_metrics.setdefault("global AUC", auc)
di_metrics.setdefault("global NDCG", ndcg)

### 4. Global accuracy (on known ratings)
y_val = (dataset_val.folds.toarray()*dataset_val.ratings.toarray()).ravel()
y_pred = predictions.toarray().ravel()
acc = [int(y==y_pred[iy]) for iy, y in enumerate(y_val) if (y!=0)]
di_metrics.setdefault("ACC", np.sum(acc)/len(acc))

gc.collect()
metrics = pd.DataFrame({("%d_%s" % (inn+1,model_name)): di_metrics})
```

### 4.b Reproduction instructions: benchmark

```bash
## Folder where files are saved
SAVE_FOLDER="../benchmark-results/" 
## Algorithms to test
ALGOS="ALSWR,LibMF,LogisticMF,PMF,SCPMF,FastaiCollabWrapper,NIMCGCN,BNNR,DRRS,HAN,LRSSL" 
DATAS=("Cdataset" "Gottlieb" "LRSSL" "DNdataset" "PREDICT" "PREDICT_Gottlieb" "Synthetic" "TRANSCRIPT")

## Note:
## "Gottlieb" -> "Fdataset" in the paper
## "PREDICT_Gottlieb" -> "Gottlieb" in the paper

SPLITS=("random_simple" "weakly_correlated")
NJOBS=1 ## Number of parallel jobs
N=100 ## Number of runs
K=5 ## Number of folds in cross-validation

for SPLIT in "${SPLITS[@]}"
do
    for DATA in "${DATAS[@]}"
    do
        echo $SPLIT"----"$DATA;
        python3 -m main --models "$ALGOS" --datasets "$DATA" --njobs "$NJOBS" --N "$N" --K "$K" --splitting "$SPLIT" --save_folder "$SAVE_FOLDER";
    done
done
```

### 4.c Reproduction instructions: statistical analyses

After the benchmark, run

```bash
python3 -m analyses
```
