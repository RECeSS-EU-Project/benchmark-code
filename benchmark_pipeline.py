#coding:utf-8

import benchscofi
import benchscofi.prior_estimation
import stanscofi
import stanscofi.datasets
import stanscofi.utils
import stanscofi.training_testing
import stanscofi.validation
import seaborn as sns
import numpy as np
import random
from multiprocessing import cpu_count
from subprocess import Popen
import gc
from joblib import Parallel, delayed
import os
import pandas as pd
from time import time

datasets_folder="datasets/"
models = [ ## 18
	"PMF", "PulearnWrapper", "FastaiCollabWrapper", "NIMCGCN", "FFMWrapper", 
	"VariationalWrapper", "DRRS", "SCPMF", "BNNR", "LRSSL", "MBiRW", "LibMFWrapper", 
	"LogisticMF", "PSGCN", "DDA_SKF", "HAN"
]
datasets = ["Synthetic", "CaseControl", "Censoring", "TRANSCRIPT", "Gottlieb", "Cdataset", "PREDICT", "LRSSL"] ## 9
splitting_methods = ["weakly_correlated", "random_simple"]

def aux_run_pipeline(inn, model_name, data_args, red_folds, splitting, random_seed, metric="AUC", K=5, ptest=0.2):
        np.random.seed(random_seed)
        random.seed(random_seed)
	dataset = stanscofi.datasets.Dataset(**data_args)
	if (red_folds is not None):
		dataset = dataset.subset(red_folds)
        ##--------------------------------------------------------------##
        ##           I. SPLIT DATASET USING splitting                   ##
	##--------------------------------------------------------------##
        (traintest_folds, val_folds), _ = eval("stanscofi.training_testing."+splitting+"_split")(dataset, ptest, metric="euclidean", random_state=random_seed)
        ##--------------------------------------------------------------##
        ##           II. TRAINING/TESTING model CV ON traintest dataset ##
	##--------------------------------------------------------------##
        dataset_traintest = dataset.subset(traintest_folds)
        model = eval("benchscofi."+model+"."+model+"()")
	start_time = time()
	di_results = stanscofi.training_testing.cv_training(model, None, dataset_traintest, K, metric, k=1, beta=1, threshold=0, cv_type="random", random_state=random_seed)
	runtime = time()-start_time
	print("Training Time %f\tBest Test %s=%f" % (runtime, metric, np.max(di_results["test_metric"])))
	## Return best model based on highest test metric
	best_model = di_results["models"][np.argmax(di_results["test_metric"])]
        ##--------------------------------------------------------------##
        ##           III. VALIDATION model ON val dataset               ##
	##--------------------------------------------------------------##
	dataset_val = dataset.subset(val_folds)
	p_start_time = time()
        scores = best_model.predict_proba(val_dataset)
	p_runtime = time()-p_start_time
	print("Predicting Time %f" % p_runtime)
        best_model.print_scores(scores)
        predictions = best_model.predict(scores, threshold=0)
        best_model.print_classification(predictions)
	## Compute metrics
	### 1. row-wise AUC (using scikit-learn) and row-wise NDCG@#items
        metrics, _ = stanscofi.validation.compute_metrics(scores, predictions, val_dataset, metrics=["AUC", "NDCGk"], k=val_dataset.nitems, beta=1, verbose=False)
	di_metrics = metrics.iloc[:-1,:].to_dict()["Average"]
	print(di_metrics)
	### 2. row-wise disagreeing AUC
        from benchscofi.utils import rowwise_metrics
        lin_auc = np.mean(rowwise_metrics.calc_auc(scores, val_dataset, transpose=False, verbose=False))
        di_metrics.setdefault("Lin's AUC", lin_auc)
	### 3. global AUC and global NDCG@#pairs and Hit Ratio @2, @5, @10
        from stanscofi.validation import AUC, NDCGk, HRk
        y_val = (val_dataset.folds.toarray()*val_dataset.ratings.toarray()).ravel()
        y_val[y_val<1] = 0
        y_pred = scores.toarray().ravel()
	auc = AUC(y_val, y_pred, 1, 1)
	ndcg = NDCGk(y_val, y_pred, y_pred.shape[0], 1)
	hk2 = HRk(y_val, y_pred, 2, 1)
	hk5 = HRk(y_val, y_pred, 5, 1)
	hk10 = HRk(y_val, y_pred, 10, 1)
        di_metrics.setdefault("global AUC", auc)
        di_metrics.setdefault("global NDCG", ndcg)
        di_metrics.setdefault("HR@2", hk2)
        di_metrics.setdefault("HR@5", hk5)
        di_metrics.setdefault("HR@10", hk10)
	### 4. Global accuracy (on known ratings)
        y_val = (val_dataset.folds.toarray()*val_dataset.ratings.toarray()).ravel()
        y_pred = predictions.toarray().ravel()
        acc = [int(y==y_pred[iy]) for iy, y in enumerate(y_val) if (y!=0)]
        di_metrics.setdefault("ACC", np.sum(acc)/len(acc))
	print(di_metrics)
        gc.collect()
	return pd.DataFrame({("%d_%s" % (inn+1,model_name)): di_metrics})

def run_pipeline(model_name, dataset_name, splitting, metric="AUC", batch_ratio=1., N=100, K=5, ptest=0.2, njobs=1):
	assert batch_ratio<=1 and batch_ratio>0
	assert splitting in splitting_methods
	assert dataset in datasets
	assert model_name in models
	assert K>=2
	assert N>0
	assert ptest<1 and ptest>0
	assert njobs>0 and njobs<cpu_count()
	results_fname = "_N=%d_%s_%s_%s_%s_%d_%f.csv" % (N, model_name, dataset_name, splitting, metric, K, ptest)
	seeds = np.random.choice(range(int(1e8)), size=N)
	if (os.path.exists("results"+results_fname) and os.path.exists("seeds"+results_fname)):
		return pd.read_csv("results"+results_fname, index=0)
        ##################################################################
        ##            IMPORT/CREATE DATASET                             ##
	##################################################################
        dataset_seed = 1234
        npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
        pi, sparsity, imbalance, c = 0.3, 0.01, 0.03, 0.3
        if (dataset_name=="Synthetic"):
        	data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=dataset_seed)
        elif (dataset_name=="CaseControl"):
        	data_args = benchscofi.prior_estimation.generate_CaseControl_dataset(N=npositive+nnegative,nfeatures=nfeatures,pi=pi,sparsity=sparsity,imbalance=imbalance,mean=mean,std=std,exact=True,random_state=dataset_seed)
        elif (dataset_name=="Censoring"):
        	data_args = benchscofi.prior_estimation.generate_Censoring_dataset(pi=pi,c=c,N=npositive+nnegative,nfeatures=nfeatures,mean=mean,std=std,exact=True,random_state=dataset_seed)
        else:
        	Popen("mkdir", "-p", datasets_folder)
        	data_args = stanscofi.utils.load_dataset(dataset_name, datasets_folder)
        dataset = stanscofi.datasets.Dataset(**data_args)
        ##################################################################
        ##            (optional) REDUCE DATASET                         ##
	##################################################################
        if (batch_ratio<1):
		dataset = stanscofi.datasets.Dataset(**data_args)
		print("Random batch of size %d (ratio=%f perc.)" % (batch_ratio*dataset.nitems*dataset.nusers, batch_ratio))
		(_, red_folds), _ = stanscofi.training_testing.random_simple_split(dataset, batch_ratio, metric="euclidean", random_state=random_seed)
	else:
		red_folds = None
	if (njobs==1):
		results = []
		for iss, seed in enumerate(seeds):
			df_results = aux_run_pipeline(iss, model_name, data_args, red_folds, splitting, seed, metric=metric, K=K, ptest=ptest)
			results.append(df_results)
	else:
		results = Parallel(n_jobs=njobs, backend='loky')(delayed(aux_run_pipeline)(iss, model_name, dataset, splitting, seed, metric=metric, K=K, ptest=ptest) for iss, seed in enumerate(seeds))
	res_df = pd.concat(tuple(results), axis=0)
	res_df.to_csv("results"+results_fname)
	pd.DataFrame([seeds], index=["seed"], columns=range(N)).to_csv("seeds"+results_fname)

def plot_boxplots(model_name, dataset_name, splitting, metric="AUC", batch_ratio=1., N=100, K=5, ptest=0.2, njobs=1):
	results_fname = "_N=%d_%s_%s_%s_%s_%d_%f.csv" % (N, model_name, dataset_name, splitting, metric, K, ptest)
	seeds = np.random.choice(range(int(1e8)), size=N)
	assert (os.path.exists("results"+results_fname) and os.path.exists("seeds"+results_fname))
	## metrics x number of iterations (=N)
	metrics = pd.read_csv("results"+results_fname, index=0)
	print(metrics) # plot figures
	metrics.boxplot()

if __name__=="__main__":
	run_pipeline("LibMFWrapper", "Gottlieb", "random_simple", metric="AUC", batch_ratio=1., N=1, K=5, ptest=0.2, njobs=1)