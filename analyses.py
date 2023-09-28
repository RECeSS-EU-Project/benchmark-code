#coding:utf-8

import pandas as pd
from benchmark_pipeline import plot_boxplots
from glob import glob
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import kruskal #, mannwhitneyu, alexandergovern

run = ["boxplots", "metric", "use_features", "use_features_synthetic", "challenge","approx_error", "compare_approx", "gen_error", "compare_gen"]

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
	[["textM", "textM", "Biol", "Biol", "Biol", "Synthetic", "Biol", "Synthetic"]]
, columns=["Gottlieb","Cdataset","TRANSCRIPT","PREDICTpublic","LRSSL","Synthetic","PREDICTGottlieb","Synthetic-wo-features"]
, index=["type"]).T
rename_datasets = {
	"Gottlieb": "Fdataset",
	"PREDICTpublic": "PREDICT",
	"PREDICTGottlieb": "Gottlieb",
}
algorithm_df = pd.DataFrame(
	[["No","No","Yes","No","No","Yes","No"],
	["MF","NN","NN","MF","MF","NN","MF"]]
, columns=["ALSWR", "FastaiCollabWrapper", "HAN", "LibMF", "LogisticMF", "NIMCGCN", "PMF"]
, index=["features", "type"]).T
metric_of_choice = "Lin's AUC"
topN=2

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

	corrmat = df_metrics.T.corr(method="spearman").values
	r2mat = np.eye(corrmat.shape[0])
	for im1, m1 in enumerate(df_metrics.index):
		for im2, m2 in enumerate(df_metrics.index[(im1+1):]):
			r2mat[im1+1+im2,im1] = r2_score(df_metrics.loc[m1].values, df_metrics.loc[m2].values)
			r2mat[im1,im1+1+im2] = r2mat[im1+1+im2,im1]

	fontsize=1.5
	sns.set(font_scale=fontsize)

	cg = sns.PairGrid(df_metrics.T, diag_sharey=False, corner=False, height=1.5, aspect=1)
	cg.fig.suptitle(r"Correlogram: $R^2$ plot & Spearman's $\rho$ (N=%d/metric)" % df_metrics.shape[1], fontsize=fontsize*10-2)
	cg.map_upper(sns.kdeplot, alpha=0)
	cg.map_lower(sns.regplot, scatter_kws = {"color": "black", "alpha": 0.2, "s": 0.1}, line_kws = {"color": "red"})
	cg.map_diag(sns.kdeplot, color="red")
	for ax, [im1,im2,metric_nms] in zip(cg.axes.flat, [[ii,jj,[i,j]] for ii, i in enumerate(list(df_metrics.index)) for jj, j in enumerate(list(df_metrics.index))]):
		min1 = np.min(df_metrics.loc[metric_nms[0]])
		max2 = np.max(df_metrics.loc[metric_nms[1]])
		std1 = np.std(df_metrics.loc[metric_nms[0]])
		std2 = np.std(df_metrics.loc[metric_nms[1]])
		if (im1<im2):
			y1, y2 = ax.get_ylim()
			x1, x2 = ax.get_xlim()
			ax.text(1/(3*(1+int(im2!=2)))*(x2+x1), 1/2*(y2+y1), "%.2f" % corrmat[im1,im2], fontsize=fontsize*15, color="black")
			ax.grid(False)
		elif (im1>im2 and r2mat[im1,im2]>0):
			ax.text(min1-0.5*std1, max2+0.5*std2, r"$R^2=$%.1f" % r2mat[im1,im2], fontsize=fontsize*15, color="red")
		else:
			pass
	plt.show()

################################
## Use features in datasets?  ##
################################

if ("use_features" in run):
	dfs_metrics = {}
	for dataset_name in dataset_df.index:
		fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		dfs_metrics.setdefault(dataset_name, {f: data_df[[m for m in data_df.columns if ((algorithm_df.loc[m]["features"]==f))]] for f in algorithm_df["features"].unique()})
		#dfs_metrics.setdefault(dataset_name, {f: data_df[[m for m in data_df.columns if ((algorithm_df.loc[m]["features"]==f) and (algorithm_df.loc[m]["type"]=="NN"))]] for f in algorithm_df["features"].unique()})

	alpha, alpha2 = 0.10, 0.01
	results = {}
	for dataset_name in dfs_metrics:
		with_features, wo_features = [dfs_metrics[dataset_name][f].values.ravel() for f in ["Yes","No"]]
		res = kruskal(with_features, wo_features)
		results.setdefault(dataset_name, {"statistic": np.round(res.statistic,2), "sign.": "*"*int(res.pvalue<alpha)+"**"*int(res.pvalue<alpha2), "mean(wf)-mean(wof)": np.mean(with_features)-np.mean(wo_features)})
	print("\n* Kruskal-Wallis H-test\n\tH_0: mean(score)_{with features}=mean(score)_{w/o features}\n\tN=%d with features\tN=%d w/o features\talpha=%.2f (%.2f)\n" % (len(with_features), len(wo_features), alpha, alpha2))
	results = pd.DataFrame(results)
	results.columns = [rename_datasets.get(x,x) for x in results.columns]
	print(results[list(sorted(results.columns))].loc[["statistic","sign.","mean(wf)-mean(wof)"]])

if ("use_features_synthetic" in run):
	dfs_metrics = {}
	for dataset_name in dataset_df.index:
		if ("Synthetic" not in dataset_name):
			continue
		fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		dfs_metrics.setdefault(dataset_name, data_df[[m for m in data_df.columns if ((algorithm_df.loc[m]["features"]=="Yes"))]])
		#dfs_metrics.setdefault(dataset_name, {f: data_df[[m for m in data_df.columns if ((algorithm_df.loc[m]["features"]==f) and (algorithm_df.loc[m]["type"]=="NN"))]] for f in algorithm_df["features"].unique()})

	alpha, alpha2 = 0.10, 0.01
	results = {}
	with_features, wo_features = [dfs_metrics["Synthetic"+d].values.ravel() for d in ["","-wo-features"]]
	res = kruskal(with_features, wo_features)
	results.setdefault("Synthetic w/wo features", {"statistic": np.round(res.statistic,2), "sign.": "*"*int(res.pvalue<alpha)+"**"*int(res.pvalue<alpha2), "mean(wf)-mean(wof)": np.median(with_features)-np.median(wo_features)})
	print("\n* Kruskal-Wallis H-test\n\tH_0: mean(score)_{with features}=mean(score)_{w/o features}\n\tN=%d with features\tN=%d w/o features\talpha=%.2f (%.2f)\n" % (len(with_features), len(wo_features), alpha, alpha2))
	results = pd.DataFrame(results)
	print(results[list(sorted(results.columns))].loc[["statistic","sign.","mean(wf)-mean(wof)"]])

################################
## Most challenging dataset?  ##
################################

if ("challenge" in run):
	dfs_metrics = {}
	for dataset_name in dataset_df.index:
		fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		dfs_metrics.setdefault(dataset_name, data_df.mean(axis=0).max())

	print(pd.DataFrame({"Rank":dfs_metrics}).sort_values(by="Rank",ascending=False).T)

####################################
## Best algorithm (approx error)  ##
####################################

## boxplots
if ("approx_error" in run):
	df_metrics = {}
	fontsize=30
	for dataset_name in dataset_df.index:
		if ("Synthetic" in dataset_name):
			continue
		fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		rank_data_df = data_df.mean(axis=0).sort_values(ascending=False)
		df_metrics.setdefault(dataset_name, data_df[list(rank_data_df.index[:topN])])

	fig, axes = plt.subplots(nrows=1,ncols=len(df_metrics),figsize=(22,2.5))
	order = ["Cdataset","Gottlieb","PREDICTGottlieb","LRSSL","PREDICTpublic","TRANSCRIPT"]
	for i, [ax, dataset_name] in enumerate(zip(axes,order)):
		sns.boxplot(data=df_metrics[dataset_name], ax=ax)
		ax.set_xticklabels(df_metrics[dataset_name].columns, rotation=45, fontsize=fontsize)
		ax.set_ylim((0.45, 0.87))
		if (i!=0):
			ax.set_yticklabels([])
		else:
			ax.set_yticklabels(ax.get_yticklabels(),fontsize=fontsize)		
		ax.set_title(rename_datasets.get(dataset_name, dataset_name), fontsize=fontsize)
	plt.savefig("boxplot_approx_error.png", bbox_inches="tight")

####################################
## Compare types (approx error)   ##
####################################

if ("compare_approx" in run):
	dfs_metrics = {}
	for dataset_name in dataset_df.index:
		fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		dfs_metrics.setdefault(dataset_name, {f: data_df[[m for m in data_df.columns if ((algorithm_df.loc[m]["type"]==f))]] for f in algorithm_df["type"].unique()})

	alpha, alpha2 = 0.10, 0.01
	results = {}
	for dataset_name in dfs_metrics:
		with_features, wo_features = [dfs_metrics[dataset_name][f].values.ravel() for f in ["MF","NN"]]
		res = kruskal(with_features, wo_features)
		results.setdefault(dataset_name, {"statistic": np.round(res.statistic,2), "sign.": "*"*int(res.pvalue<alpha)+"**"*int(res.pvalue<alpha2), "mean(NN)-mean(MF)": np.mean(wo_features)-np.mean(with_features)})
	print("\n* Kruskal-Wallis H-test\n\tH_0: mean(score)_{NN}=mean(score)_{MF}\n\tN=%d (MF)\tN=%d (NN)\talpha=%.2f (%.2f)\n" % (len(with_features), len(wo_features), alpha, alpha2))
	results = pd.DataFrame(results)
	results.columns = [rename_datasets.get(x,x) for x in results.columns]
	print(results[list(sorted(results.columns))].loc[["statistic","sign.","mean(NN)-mean(MF)"]])

####################################
## Best algorithm (gen error)     ##
####################################

## boxplots
if ("gen_error" in run):
	df_metrics = {}
	fontsize=30
	for dataset_name in dataset_df.index:
		if ("Synthetic" in dataset_name):
			continue
		fnames = glob(root_folder+"results_%s_weakly_correlated/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		rank_data_df = data_df.mean(axis=0).sort_values(ascending=False)
		df_metrics.setdefault(dataset_name, data_df[list(rank_data_df.index[:topN])])

	fig, axes = plt.subplots(nrows=1,ncols=len(df_metrics),figsize=(22,2.5))
	order = ["Cdataset","Gottlieb","PREDICTGottlieb","LRSSL","PREDICTpublic","TRANSCRIPT"]
	for i, [ax, dataset_name] in enumerate(zip(axes,order)):
		sns.boxplot(data=df_metrics[dataset_name], ax=ax)
		ax.set_xticklabels(df_metrics[dataset_name].columns, rotation=45, fontsize=fontsize)
		ax.set_ylim((0.45, 0.87))
		if (i!=0):
			ax.set_yticklabels([])
		else:
			ax.set_yticklabels(ax.get_yticklabels(),fontsize=fontsize)		
		ax.set_title(rename_datasets.get(dataset_name, dataset_name), fontsize=fontsize)
	plt.savefig("boxplot_gen_error.png", bbox_inches="tight")

####################################
## Compare types (gen error)      ##
####################################

if ("compare_gen" in run):
	dfs_metrics = {}
	for dataset_name in dataset_df.index:
		fnames = glob(root_folder+"results_%s_weakly_correlated/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		dfs_metrics.setdefault(dataset_name, {f: data_df[[m for m in data_df.columns if ((algorithm_df.loc[m]["type"]==f))]] for f in algorithm_df["type"].unique()})

	alpha, alpha2 = 0.10, 0.01
	results = {}
	for dataset_name in dfs_metrics:
		if (dataset_name == "Synthetic-wo-features"):
			continue
		with_features, wo_features = [dfs_metrics[dataset_name][f].values.ravel() for f in ["MF","NN"]]
		res = kruskal(with_features, wo_features)
		results.setdefault(dataset_name, {"statistic": np.round(res.statistic,2), "sign.": "*"*int(res.pvalue<alpha)+"**"*int(res.pvalue<alpha2), "mean(NN)-mean(MF)": np.mean(wo_features)-np.mean(with_features)})
	print("\n* Kruskal-Wallis H-test\n\tH_0: mean(score)_{NN}=mean(score)_{MF}\n\tN=%d (MF)\tN=%d (NN)\talpha=%.2f (%.2f)\n" % (len(with_features), len(wo_features), alpha, alpha2))
	results = pd.DataFrame(results)
	results.columns = [rename_datasets.get(x,x) for x in results.columns]
	print(results[list(sorted(results.columns))].loc[["statistic","sign.","mean(NN)-mean(MF)"]])

####################################
## List all results               ##
####################################

if ("list-all-results-random-simple" in run):
	dfs_metrics = {}
	for dataset_name in dataset_df.index:
		fnames = glob(root_folder+"results_%s/results_*/results_*.csv" % dataset_name)
		results_di = {fnn.split("/")[-2].split("_")[1]: pd.read_csv(fnn, index_col=0) for fnn in fnames if (fnn.split("/")[-2].split("_")[1] in algorithm_df.index)} ## all iterations
		for x in results_di:
			results_di[x].columns = range(results_di[x].shape[1]) # N
		data_df = pd.DataFrame({model: results_di[model].loc[metric_of_choice].to_dict() for model in results_di})
		data_df = pd.DataFrame(data_df)
		dfs_metrics.setdefault(dataset_name, data_df)

	print(dfs_metrics)
