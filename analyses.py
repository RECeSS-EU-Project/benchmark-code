#coding:utf-8

import pandas as pd
from benchmark_pipeline import plot_boxplots
from glob import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

run = [""]

###############
## Boxplots  ##
###############

root_folder="../benchmark-results/"

for dataset_name in list(set([x.split("_")[1] for x in glob(root_folder+"results_*")])):
	if ("boxplots" not in run):
		break
	print("* %s " % dataset_name)
	fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
	results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames}
	data_df = pd.DataFrame({model: results_di[model].loc[["AUC","global AUC","Lin's AUC"]+["NDCGk","global NDCG"]].mean(axis=1).to_dict() for model in results_di})
	data_df = np.round(data_df, 3)
	print(data_df)
	data_df.to_csv("../images/results_%s.txt" % dataset_name, sep="&")
	plot_boxplots(results_di, "random_simple", dataset_name, metrics=["AUC","global AUC","Lin's AUC"], results_folder="../images/")
	plot_boxplots(results_di, "random_simple", dataset_name, metrics=["NDCGk","global NDCG"], results_folder="../images/")
	fnames = glob(root_folder+"results_%s_weakly_correlated/results_*/results_*.csv" % dataset_name)
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

################################
## Comparing metrics          ##
################################

if ("metric" in run):
	dfs_metrics = []
	for dataset_name in dataset_df.index:
		fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
		results_di = [pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)]
		for ix, x in enumerate(results_di):
			results_di[ix].columns = [col+"_"+dataset_name for col in list(x.columns)]
		dfs_metrics += results_di

	df_metrics = dfs_metrics[0].join(dfs_metrics[1:], how="outer")
	df_metrics = df_metrics.loc[[i for i in df_metrics.index if (" time (sec)" not in i)]]
	df_metrics = df_metrics.loc[["ACC", "global AUC", "AUC", "Lin's AUC", "global NDCG"]]
	df_metrics.index = [{
		"ACC": "Accuracy", 
		"global AUC": "AUC",
		"AUC": "AUC/disease",
		"HR@2": "Recall@2",
		"HR@10": "Recall@10",
		"HR@5": "Recall@5",
		"Fscore": "F1/disease",
		"NDCGk": "NDCG/disease",
		"Lin's AUC": "NS AUC",
		"global NDCG": "NDCG"
	}.get(x, x) for x in df_metrics.index]

	## TODO redo!!!
	corrmat = df_metrics.T.corr(method="spearman").values
	#corrmat = np.triu(corrmat, k=-1)
	corrmat = pd.DataFrame(corrmat, index=df_metrics.index, columns=df_metrics.index)
	r2mat = np.zeros(corrmat.shape).astype(int).astype(str)
	r2mat[r2mat=='0'] = ""
	for im1, m1 in enumerate(df_metrics.index):
		r2mat[im1,im1] = "1.0"
		for im2, m2 in enumerate(df_metrics.index[(im1+1):]):
			r2mat[im1+1+im2,im1] = r"$R^2$="+str(np.round(r2_score(df_metrics.loc[m1].values, df_metrics.loc[m2].values),1))
			r2mat[im1,im1+1+im2] = r"$\rho$="+str(np.round(corrmat.values[im1,im1+1+im2], 1))

	cg = sns.clustermap(corrmat, fmt="s", figsize=(8,8), annot=r2mat, cbar_kws=None, cmap="coolwarm", vmin=-1, vmax=1)
	#cg = sns.PairGrid(df_metrics, diag_sharey=False, corner=True, height=1., aspect=1)
	#cg.map_upper(sns.kdeplot)#clustermap)
	#cg.map_lower(sns.regplot)
	#cg.map_diag(sns.histplot)
	plt.setp(cg.ax_heatmap.xaxis.get_majorticklabels(), rotation=45)
	cg.ax_col_dendrogram.set_visible(False)
	cg.fig.suptitle("Spearman correlation heatmap")
	plt.show()
	plt.close()

################################
## Use features in datasets?  ##
################################



exit()

dfs_metrics = {}
for dataset_name in dataset_df.index:
	fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
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
