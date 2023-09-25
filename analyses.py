#coding:utf-8

import pandas as pd
from benchmark_pipeline import plot_boxplots
from glob import glob
import numpy as np

run = [""]

###############
## Boxplots  ##
###############

for dataset_name in list(set([x.split("_")[1] for x in glob("results_*")])):
	if ("boxplots" not in run):
		break
	print("* %s " % dataset_name)
	fnames = glob("results_%s/results_*/results_*.csv" % dataset_name)
	results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames}
	data_df = pd.DataFrame({model: results_di[model].loc[["AUC","global AUC","Lin's AUC"]+["NDCGk","global NDCG"]].mean(axis=1).to_dict() for model in results_di})
	data_df = np.round(data_df, 3)
	print(data_df)
	data_df.to_csv("../images/results_%s.txt" % dataset_name, sep="&")
	plot_boxplots(results_di, "random_simple", dataset_name, metrics=["AUC","global AUC","Lin's AUC"], results_folder="../images/")
	plot_boxplots(results_di, "random_simple", dataset_name, metrics=["NDCGk","global NDCG"], results_folder="../images/")
	fnames = glob("results_%s_weakly_correlated/results_*/results_*.csv" % dataset_name)
	if (len(fnames)>0):
		print("* %s (weakly correlated)" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames}
		data_df = pd.DataFrame({model: results_di[model].loc[["AUC","global AUC","Lin's AUC"]+["NDCGk","global NDCG"]].mean(axis=1).to_dict() for model in results_di})
		data_df = np.round(data_df, 3)
		print(data_df)
		data_df.to_csv("../images/results_%s.txt" % dataset_name, sep="&")
		plot_boxplots(results_di, "weakly_correlated", dataset_name, metrics=["AUC","global AUC","Lin's AUC"], results_folder="../images/")
		plot_boxplots(results_di, "weakly_correlated", dataset_name, metrics=["NDCGk","global NDCG"], results_folder="../images/")

## Get metrics (random simple split)
dataset_df = pd.DataFrame(
	[["textM", "textM", "Biol", "Biol", "Biol", "Synthetic", "Biol"]]
, columns=["Gottlieb","Cdataset","TRANSCRIPT","PREDICTpublic","LRSSL","Synthetic","PREDICTGottlieb"]
, index=["type"]).T
algorithm_df = pd.DataFrame(
	[["No","No","Yes","No","No","Yes","No"],
	["MF","NN","NN","MF","MF","NN","MF"]]
, columns=["ALSWR", "FastaiCollabWrapper", "HAN", "LibMF", "LogisticMF", "NIMCGCN", "PMF"]
, index=["features", "type"]).T

metric_of_choice = "Lin's AUC"

## by metric
dfs_metrics = []
for dataset_name in dataset_df.index:
	fnames = glob("results_%s/results_*/results_*.csv" % dataset_name)
	results_di = [pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)]
	for ix, x in enumerate(results_di):
		results_di[ix].columns = [col+"_"+dataset_name for col in list(x.columns)]
	dfs_metrics += results_di

print(dfs_metrics[0].join(dfs_metrics[1:], how="outer"))

################################
## Comparing metrics          ##
################################


exit()

dfs_metrics = {}
for dataset_name in dataset_df.index:
	fnames = glob("results_%s/results_*/results_*.csv" % dataset_name)
	results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
	for x in results_di:
		results_di[x].columns = range(results_di[x].shape[1]) # N
	data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
	data_df = pd.DataFrame(data_df)
	dfs_metrics.setdefault(dataset_name, data_df)

print(dfs)

################################
## Most challenging dataset?  ##
################################
