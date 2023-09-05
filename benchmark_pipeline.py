#coding:utf-8

## TIMEOUT: https://stackoverflow.com/questions/492519/timeout-on-a-function-call
## MEMORYERROR: https://stackoverflow.com/questions/9850995/tracking-maximum-memory-usage-by-a-python-function
## EXCEPTIONHANDLING: TODO

import benchscofi
from benchscofi.utils import rowwise_metrics, prior_estimation
import stanscofi
import stanscofi.datasets
import stanscofi.utils
import stanscofi.training_testing
import stanscofi.validation
from stanscofi.validation import AUC, NDCGk, HRk
import seaborn as sns
import numpy as np
import random
from multiprocessing import cpu_count
from subprocess import Popen, call
import gc
from joblib import Parallel, delayed
import os
import pandas as pd
from time import time
import matplotlib.pyplot as plt
import json

datasets_folder="datasets/"
models = [ ## 18
	"Constant", "PMF", "PulearnWrapper", "FastaiCollabWrapper", "NIMCGCN", "FFMWrapper", 
	"VariationalWrapper", "DRRS", "SCPMF", "BNNR", "LRSSL", "MBiRW", "LibMFWrapper", 
	"LogisticMF", "PSGCN", "DDA_SKF", "HAN", "ALSWR", "LibMF", "SimpleBinaryClassifier"
]
datasets = ["Synthetic", #"CaseControl", "Censoring", 
"TRANSCRIPT", "Gottlieb", "Cdataset", "PREDICT", "LRSSL"] ## 9
splitting_methods = ["weakly_correlated", "random_simple"]

def aux_run_pipeline(inn, model_name, params, data_args, red_folds, splitting, random_seed=1234, metric="AUC", K=5, ptest=0.2, verbose=False, intermediary_results_folder="./"):
	Popen(("mkdir -p "+intermediary_results_folder).split(" "))
	np.random.seed(random_seed)
	random.seed(random_seed)
	dataset = stanscofi.datasets.Dataset(**data_args)
	intermediary_results_fname = "%s/intermediary_seed=%d_%s_%s_%s_%s_%d_%f.csv" % (intermediary_results_folder, random_seed, model_name, dataset.name, splitting, metric, K, ptest)
	if (os.path.exists(intermediary_results_fname)):
		return pd.read_csv(intermediary_results_fname, index_col=0)
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
	import benchscofi
	__import__("benchscofi."+model_name)
	model = eval("benchscofi."+model_name+"."+model_name)
	start_time = time()
	di_results = stanscofi.training_testing.cv_training(model, params, dataset_traintest, K, metric, k=1, beta=1, threshold=0, cv_type="random", random_state=random_seed)
	runtime = time()-start_time
	if (verbose):
		print("it=%d\tTraining Time %f\tBest Test %s=%f" % (inn+1, runtime, metric, np.max(di_results["test_metric"])))
	## Return best model based on highest test metric
	#print(di_results["test_metric"])
	best_model = di_results["models"][np.argmax(di_results["test_metric"])]
        ##--------------------------------------------------------------##
        ##           III. VALIDATION model ON val dataset               ##
	##--------------------------------------------------------------##
	dataset_val = dataset.subset(val_folds)
	p_start_time = time()
	scores = best_model.predict_proba(dataset_val)
	p_runtime = time()-p_start_time
	if (verbose):
		print("it=%d\tPredicting Time %f" % (inn+1,p_runtime))
	best_model.print_scores(scores)
	predictions = best_model.predict(scores, threshold=0)
	best_model.print_classification(predictions)
	## Compute metrics
	### 1. row-wise AUC (using scikit-learn) and row-wise NDCG@#items
	metrics, plot_args = stanscofi.validation.compute_metrics(scores, predictions, dataset_val, metrics=["AUC", "NDCGk", "Fscore"], k=dataset_val.nitems, beta=1, verbose=False)
	#if (verbose):
	#	stanscofi.validation.plot_metrics(**plot_args, figsize=(10,10), model_name=best_model.name+" "+dataset.name)
	di_metrics = metrics.iloc[:-1,:].to_dict()["Average"]
	di_metrics.setdefault("training time (sec)", runtime)
	di_metrics.setdefault("prediction time (sec)", p_runtime)
	### 2. row-wise disagreeing AUC
	lin_aucs = rowwise_metrics.calc_auc(scores, dataset_val, transpose=False, verbose=False)
	#print(lin_aucs)
	lin_auc = np.mean(lin_aucs) if (np.max(lin_aucs)>0) else 0.5
	di_metrics.setdefault("Lin's AUC", lin_auc)
	### 3. global AUC and global NDCG@#pairs and Hit Ratio @2, @5, @10
	y_val = (dataset_val.folds.toarray()*dataset_val.ratings.toarray()).ravel()
	y_val[y_val<1] = 0
	y_pred = scores.toarray().ravel()
	auc = AUC(y_val, y_pred, 1, 1)
	ndcg = NDCGk(y_val, y_pred, y_pred.shape[0], 1)
	def HitRatio(y_val, y_pred, k, u1):
		rec_ids = np.argsort(y_pred).tolist()
		rec_ids.reverse()
		return np.sum(y_val[rec_ids[:k]])/np.sum(y_val)
	#hk2 = HRk(y_val, y_pred, 2, 1)
	hk2 = HitRatio(y_val, y_pred, 2, 1)
	hk5 = HitRatio(y_val, y_pred, 5, 1)
	hk10 = HitRatio(y_val, y_pred, 10, 1)
	di_metrics.setdefault("global AUC", auc)
	di_metrics.setdefault("global NDCG", ndcg)
	di_metrics.setdefault("HR@2", hk2)
	di_metrics.setdefault("HR@5", hk5)
	di_metrics.setdefault("HR@10", hk10)
	### 4. Global accuracy (on known ratings)
	y_val = (dataset_val.folds.toarray()*dataset_val.ratings.toarray()).ravel()
	y_pred = predictions.toarray().ravel()
	acc = [int(y==y_pred[iy]) for iy, y in enumerate(y_val) if (y!=0)]
	di_metrics.setdefault("ACC", np.sum(acc)/len(acc))
	gc.collect()
	metrics = pd.DataFrame({("%d_%s" % (inn+1,model_name)): di_metrics})
	#print(metrics)
	metrics.to_csv(intermediary_results_fname)
	return metrics

def run_pipeline(model_name=None, dataset_name=None, splitting=None, params=None, metric="AUC", batch_ratio=1., N=100, K=5, ptest=0.2, njobs=1, dataset_seed=1234, verbose=False, datasets_folder="./", results_folder="./"):
	assert batch_ratio<=1 and batch_ratio>0
	assert splitting in splitting_methods
	assert dataset_name in datasets
	assert model_name in models
	assert K>=2
	assert N>0
	assert ptest<1 and ptest>0
	assert njobs>0 and njobs<cpu_count()
	Popen(("mkdir -p "+results_folder).split(" "))
	np.random.seed(dataset_seed)
	random.seed(dataset_seed)
	results_fname = "_%s_%s_%s_%s_%f_%d_%f.csv" % (model_name, dataset_name, splitting, metric, batch_ratio, K, ptest)
	seeds = np.random.choice(range(int(1e8)), size=N)
	#print(seeds)
	if (os.path.exists(results_folder+("results_N=%d" % N)+results_fname) and os.path.exists(results_folder+("seeds_N=%d" % N)+results_fname)):
		#print(pd.read_csv(("seeds_N=%d" % N)+results_fname, index_col=0))
		return pd.read_csv(results_folder+("results_N=%d" % N)+results_fname, index_col=0)
        ##################################################################
        ##            IMPORT/CREATE DATASET                             ##
	##################################################################
	npositive, nnegative, nfeatures, mean, std = 200, 100, 50, 0.5, 1
	pi, sparsity, imbalance, c = 0.3, 0.01, 0.03, 0.3
	if (dataset_name=="Synthetic"):
		data_args = stanscofi.datasets.generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=dataset_seed)
		data_args.setdefault("name", "Synthetic")
	#elif (dataset_name=="CaseControl"):
	#	data_args, _ = prior_estimation.generate_CaseControl_dataset(N=npositive+nnegative,nfeatures=nfeatures,pi=pi,sparsity=sparsity,imbalance=imbalance,mean=mean,std=std,exact=True,random_state=dataset_seed)
	#	data_args.setdefault("name", "CaseControl")
	#elif (dataset_name=="Censoring"):
	#	data_args, _ = prior_estimation.generate_Censoring_dataset(pi=pi,c=c,N=npositive+nnegative,nfeatures=nfeatures,mean=mean,std=std,exact=True,random_state=dataset_seed)
	#	data_args.setdefault("name", "Censoring")
	else:
		Popen(("mkdir -p "+datasets_folder).split(" "))
		data_args = stanscofi.utils.load_dataset(dataset_name, datasets_folder)
	dataset = stanscofi.datasets.Dataset(**data_args)
        ##################################################################
        ##            (optional) REDUCE DATASET                         ##
	##################################################################
	if (batch_ratio<1):
		dataset = stanscofi.datasets.Dataset(**data_args)
		if (verbose):
			print("Random batch of size %d (ratio=%f perc.)" % (batch_ratio*dataset.nitems*dataset.nusers, batch_ratio))
		(_, red_folds), _ = stanscofi.training_testing.random_simple_split(dataset, batch_ratio, metric="euclidean", random_state=random_seed)
	else:
		red_folds = None
        ##################################################################
        ##            RUN BENCHMARK PIPELINE                            ##
	##################################################################
	if ((njobs==1) or (N==1)):
		results = []
		for iss, seed in enumerate(seeds):
			df_results = aux_run_pipeline(iss, model_name, params, data_args, red_folds, splitting, random_seed=seed, metric=metric, K=K, ptest=ptest, verbose=verbose, intermediary_results_folder=results_folder)
			results.append(df_results)
	else:
		if (verbose):
			print("%d jobs in parallel" % njobs)
		results = Parallel(n_jobs=njobs, backend='loky')(delayed(aux_run_pipeline)(iss, model_name, params, data_args, red_folds, splitting, random_seed=seed, metric=metric, K=K, ptest=ptest, verbose=verbose, intermediary_results_folder=results_folder) for iss, seed in enumerate(seeds))
	res_df = pd.concat(tuple(results), axis=1)
	res_df.to_csv((results_folder+("results_N=%d" % N))+results_fname)
	pd.DataFrame([seeds], index=["seed"], columns=range(N)).to_csv((results_folder+("seeds_N=%d" % N))+results_fname)
	#Popen(("rm -f intermediary_seed=*_"+results_fname).split(" "), shell=True)
	call((("rm -f %s/intermediary_seed=*_" % results_folder)+results_fname), shell=True)
	return res_df

def plot_boxplots(results_di, splitting, dataset_name, metrics=None, results_folder="./"):
	if (metrics is None or len(results_di)==1):
		for model_name in results_di:
			ids = [i for i in results_di[model_name].index if ((" time (sec)" not in i) and ("HR@" not in i) and (i not in ["ACC", "Fscore"]))]
			results = results_di[model_name].loc[ids]
			sns.boxplot(data=results.T)
			plt.xlabel("Metrics (%s)" % model_name)
			plt.ylabel("Score")
			plt.xticks(rotation=45)
			plt.savefig("%s/boxplot_%s_%s_%s.png" % (results_folder,model_name,splitting,dataset_name), bbox_inches="tight")
			plt.close()
	else:
		results_lst = []
		for metric in metrics:
			results = pd.concat(tuple([pd.DataFrame(results_di[i].loc[metric].values, columns=["value"], index=["%s (%d)" % (i,j) for j in range(results_di[i].shape[1])]) for i in results_di]), axis=0)
			results["model"] = [ii for i in results_di for ii in [i]*results_di[i].shape[1]]
			results["metric"] = [metric]*results.shape[0]
			results_lst.append(results.T)
		results = pd.concat(tuple(results_lst), axis=1).T
		results["value"] = results["value"].astype(float)
		sns.boxplot(data=results, x="value", y="metric", hue="model")
		plt.xlabel("Score")
		plt.ylabel("Metric")
		plt.xticks(rotation=45)
		plt.savefig("%s/boxplot_%s_%s_%s.png" % (results_folder,model_name,splitting,dataset_name), bbox_inches="tight")
		plt.close()

if __name__=="__main__":
	from multiprocessing import cpu_count

	N = 3
	params_all = {
		"model_name" : "PMF",
		"dataset_name" : "Synthetic",
		"splitting" : "random_simple",
		"params" : None,
		"metric" : "AUC",
		"batch_ratio" : 1.,
		"N" : N, # nb iterations
		"K" : 5, # nb folds
		"ptest" : 0.2, # size of testing set
		"njobs" : max(1,min(N,cpu_count()-2)), # parallelism
		"verbose" : True,
		"results_folder" : "results_test/",
		"datasets_folder" : "datasets/",
	}

	proc = Popen(("mkdir -p "+params_all["results_folder"]).split(" "))
	proc.wait()
	with open(params_all["results_folder"]+"/params_"+"_".join([p+"="+str(v) for p, v in params_all.items() if (p not in ["params", "results_folder", "datasets_folder"])])+".json", "w") as f:
		f.write(json.dumps(params_all))

	results = run_pipeline(**params_all)
	print(results)
	plot_boxplots({params_all["model_name"]: results}, params_all["splitting"], params_all["dataset_name"], metrics=None,  results_folder=params_all["results_folder"])
	plot_boxplots({params_all["model_name"]: results, params_all["model_name"]+"_2": results}, params_all["splitting"], params_all["dataset_name"], metrics=["AUC"], results_folder=params_all["results_folder"])
	plot_boxplots({params_all["model_name"]: results, params_all["model_name"]+"_2": results}, params_all["splitting"], params_all["dataset_name"], metrics=["AUC","Lin's AUC"], results_folder=params_all["results_folder"])
	#Popen(("rm -rf %s" % params_all["results_folder"]).split(" "), shell=True)
	call("rm -rf %s" % params_all["results_folder"], shell=True)