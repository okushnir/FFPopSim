import sys
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import scipy
import numpy as np
from matplotlib import gridspec
import math

sns.set_context("paper")
sns.set_style("whitegrid")


def plot_line(**kwargs):
    plt.axvline(0, linestyle="--", color="black")
                
def main(args):
    data_1_10=pd.read_table("../hiv_fitness/inferences_replication_1_10.txt", sep=" ", index_col=False, names=["Pos", "Fitness"])
    data_10_30=pd.read_table("../hiv_fitness/inferences_replication_10_30.txt", sep=" ", index_col=False, names=["Pos", "Fitness"])
    data_1_10_uniform=pd.read_table("../hiv_fitness/inferences_replication_uniform_1_10.txt", sep=" ", index_col=False, names=["Pos", "Fitness"])
    data_10_30_uniform=pd.read_table("../hiv_fitness/inferences_replication_uniform_10_30.txt", sep=" ", index_col=False, names=["Pos", "Fitness"])

    log_coeffs = np.load("../hiv_fitness/replicative_fitness_coefficients.npy")
    e_coeffs=np.exp(log_coeffs)
    one_mins_coeffs=np.subtract(log_coeffs,-1)
    coeffs_df=pd.DataFrame(e_coeffs, columns=["Fitness"])
    coeffs_df["Pos"]=range(1,10001)

    simulated="Simulated"
    first="1-10, smoothed"
    firstUniform="1-10, uniform"
    last="10,20,30, smoothed"
    lastUniform="10-30, uniform"
    
    data_1_10_uniform["Generations"]=firstUniform
    data_10_30_uniform["Generations"]=lastUniform
    data_1_10["Generations"]=first
    data_10_30["Generations"]=last
    coeffs_df["Generations"]=simulated

    data=pd.concat([data_1_10, data_10_30, data_1_10_uniform, data_10_30_uniform])
    
    sns.distplot(data_1_10["Fitness"], kde=True, kde_kws={"label": first}, hist=False)
    sns.distplot(data_10_30["Fitness"], kde=True, kde_kws={"label": last}, hist=False)
    sns.distplot(data_1_10_uniform["Fitness"], kde=True, kde_kws={"label": firstUniform}, hist=False)
    sns.distplot(data_10_30_uniform["Fitness"], kde=True, kde_kws={"label": lastUniform}, hist=False)
    sns.distplot(e_coeffs, kde=True, kde_kws={"label":simulated}, hist=False)
    plt.savefig("../figures/kdes.png", dpi=300)
    plt.close()
    data=data.merge(coeffs_df[["Pos","Fitness"]], how="inner", on="Pos", suffixes=("_inferred", "_simulated"))   

    ax=sns.relplot("Fitness_simulated", "Fitness_inferred", data=data, col="Generations", col_wrap=2, alpha=.05)
    ax.set(yscale="symlog")
    ax.set(xscale="symlog")    
    ax.set(xlim=(0,1.3))
    ax.set(ylim=(0,1.3))
    #ax.map(plt.plot,x=[0, 1.5], y=[0, 1.5], linewidth=2, color=".3")#color="#4e7496")  
    ax.map_dataframe(plt.plot, [0, 1.3], [0, 1.3], 'r-', color="black")
    ax.set(xlabel="Simulated fitness")
    ax.set(ylabel="Inferred fitness")
    ax.set(xticks=np.arange(0, 1.31, 0.2))
    ax.set(yticks=np.arange(0, 1.31, 0.2))

    plt.legend()
    plt.savefig("../figures/diag.png", dpi=300)
    plt.close()
    
    data["error"]=data["Fitness_inferred"]-data["Fitness_simulated"]
    data["class"]=np.where(data["Fitness_simulated"]>1.01, "ADV",
                    np.where(data["Fitness_simulated"]>0.99,"NEU",
                        np.where(data["Fitness_simulated"]>0.5,"DEL","LETHAL")))
    
    g = sns.FacetGrid(data=data[data["Generations"]==lastUniform], col="class", hue="class", col_order=["LETHAL","DEL","NEU","ADV"])
    g.map(sns.distplot,"error", hist=False)
    g.set(xlim=(-1,1))
    g.set(ylabel="")
    g.set(yticks=[])
    g.map(plot_line)
    #for label in ["ADV","NEU","DEL","LETHAL"]:
    #sns.distplot(data[data["class"]==label]["error"], label=label, hist=False)

    #plt.legend()
    plt.savefig("../figures/residuals.png", dpi=300)
    plt.close()
    
    g = sns.JointGrid(x="Fitness_simulated", y="Fitness_inferred", data=data[(data["Generations"]==last)])
    g = g.plot_joint(plt.scatter, edgecolor="white", alpha=0.4)
    #g = g.plot_joint(plt.scatter,
    #                 color="g", s=40, edgecolor="white", alpha=0.1)
    g = g.plot_marginals(sns.distplot, kde=False)
    rsquare = lambda a, b: stats.pearsonr(a, b)[0] ** 2
    g = g.annotate(rsquare, template="{stat}: {val:.2f}", stat="$R^2$", loc="upper left", fontsize=12)    #g.set_yscale("symlog")
    #g.set_xscale("symlog")    
    plt.show()
    
    ax=sns.pairplot(data, hue="Generations", x_vars=["Fitness_inferred","Fitness_simulated"], y_vars=["Fitness_simulated","Fitness_inferred"])
    ax.set(yscale="symlog")
    ax.set(xscale="symlog")
    ax.set(xlim=(0,1.5))
    ax.set(ylim=(0,1.5))
    #plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
    