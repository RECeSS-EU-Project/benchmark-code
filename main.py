#coding:utf-8

## FINETUNING PARAMETERS: TODO

from benchmark_pipeline import run_pipeline, plot_boxplots
import json
import argparse
from multiprocessing import cpu_count
from subprocess import Popen

parser = argparse.ArgumentParser(description='Large-scale benchmark of collaborative filtering applications to drug repurposing')

parser.add_argument('--models', type=str, help="List of models to test, separated by commas", default="PMF,FastaiCollabWrapper")
parser.add_argument('--datasets', type=str, help="List of datasets to test, separated by commas", default="Gottlieb,Synthetic")
parser.add_argument('--njobs', type=int, help="Number of parallel jobs", default=1)

parser.add_argument('--N', type=int, help="Number of iterations", default=100)
parser.add_argument('--K', type=int, help="Number of folds", default=5)
parser.add_argument('--test_size', type=float, help="Test size (in number of known ratings)", default=0.2)
parser.add_argument('--metric', type=str, help="Metric to optimize during crossvalidation", default="AUC")
parser.add_argument('--splitting', type=str, help="Type of data splitting into training/testing and validation sets", default="random_simple,weakly_correlated", choices=["random_simple", "weakly_correlated", "random_simple,weakly_correlated", "weakly_correlated,random_simple"])

parser.add_argument('--batch_ratio', type=float, help="Percentage of dataset to consider", default=1.)
parser.add_argument('--verbose', type=bool, help="Verbose", default=False)
args = parser.parse_args()

models = args.models.split(",")
datasets = args.datasets.split(",")
splitting = args.splitting.split(",")

assert args.njobs>0 and args.njobs<cpu_count()
assert args.test_size>0 and args.test_size<1
assert args.batch_ratio>0 and args.batch_ratio<=1

for model in models:
	for dataset in datasets:
		for split_ in splitting:

			params_all = {
				"model_name" : model,
				"dataset_name" : dataset,
				"splitting" : split_,
				"params" : None,
				"metric" : args.metric,
				"batch_ratio" : args.batch_ratio,
				"N" : args.N, 
				"K" : args.K, 
				"ptest" : args.test_size, 
				"njobs" : args.njobs, 
				"verbose" : args.verbose,
				"results_folder" : "results_%s/" % model,
				"datasets_folder" : "datasets/",
			}
			proc = Popen(("mkdir -p "+params_all["results_folder"]).split(" "))
			proc.wait()
			with open(params_all["results_folder"]+"/params_"+"_".join([p+"="+str(v) for p, v in params_all.items() if (p not in ["params", "results_folder", "datasets_folder"])])+".json", "w") as f:
				f.write(json.dumps(params_all))

			results = run_pipeline(**params_all)
			plot_boxplots({model: results}, params_all["splitting"], params_all["dataset_name"], metrics=None, results_folder=params_all["results_folder"])