
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import Locator
import math
from scipy import stats
import seaborn as sns
import numpy as np
from statannotations.Annotator import Annotator
import contextlib


sns.set_style("ticks")


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """
    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i-1]
            if abs(majorlocs[i-1] + majorstep/2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i-1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError('Cannot get tick locations for a '
                                  '%s type.' % type(self))


def suplabel(axis,label,label_prop=None,
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = pylab.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None:
        label_prop = dict()
    pylab.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)

def post_data_mutation(input_dir):
    mutation_lst = glob.glob(input_dir + "/*")
    columns = ["pos", "inferred_mu", "levenes_p", "filename", "Mutation"]
    df = pd.DataFrame()
    for mutation in mutation_lst:
        mutation = mutation.split("/")[-1]
        file = input_dir + "/" + str(mutation) + "/all.txt"
        data = pd.read_csv(file, sep="\t")
        data["Mutation"] = file.split("/")[-2]
        # data["label"] = file.split("/")[-7]
        df = df.append(data)
    df = pd.DataFrame(df, columns=columns)
    return df

def post_data_fitness(input_dir):
    mutation_lst = glob.glob(input_dir + "/*")
    columns = ["pos", "inferred_w", "category", "levenes_p", "filename", "Mutation"]
    df = pd.DataFrame()
    for mutation in mutation_lst:
        mutation = mutation.split("/")[-1]
        file = input_dir + "/" + str(mutation) + "/all.txt"
        data = pd.read_csv(file, sep="\t")
        data["Mutation"] = file.split("/")[-2]
        # data["label"] = file.split("/")[-7]
        df = df.append(data)
    df = pd.DataFrame(df, columns=columns)
    return df

def syn_fitness(input_dir):
    files = glob.glob(input_dir + "/posterior_fitness_syn_*")
    columns = ["distance", "allele1", "Mutation"]
    df = pd.DataFrame()
    for file in files:
        data = pd.read_csv(file, sep="\t")
        data["Mutation"] = file.split("_")[-1].split(".")[0]
        # data["label"] = file.split("/")[-7]
        df = df.append(data)
    df = pd.DataFrame(df, columns=columns)
    return df

def x_round(x):
    return math.ceil(x*10)/10

def qqplot(x, y, **kwargs):
    _, xr = stats.probplot(x, fit=False)
    _, yr = stats.probplot(y, fit=False)
    plt.scatter(xr, yr, **kwargs)


def main():
    sim_data = pd.read_table("FITS/sim_data_hiv/posterior_mutation_syn.txt", header=0)
    sim_data = sim_data[["allele0_1"]]
    sim_data["Type"] = "Simulated population sample"
    sim_data = sim_data.rename(columns={"allele0_1": "Mutation rate"})
    sim_data_noisy = pd.read_table("FITS/sim_data_hiv_noisy/posterior_mutation_syn.txt", header=0)
    sim_data_noisy = sim_data_noisy[["allele0_1"]]
    sim_data_noisy["Type"] = "Simulated population sample + sequence error"
    sim_data_noisy = sim_data_noisy.rename(columns={"allele0_1": "Mutation rate"})

    all_data = pd.concat([sim_data, sim_data_noisy], sort=False)

    all_data["Mutation rate"] = all_data["Mutation rate"].map(lambda x: str(x).lstrip('*'))
    all_data["Mutation rate"] = pd.to_numeric(all_data["Mutation rate"], errors='coerce')
    all_data["Mutation"] = "Mutation"

    #Plots
    hue_order = ["Simulated population sample", "Simulated population sample + sequence error"]
    plt.style.use('classic')
    sns.set_palette("Set2")
    g1 = sns.violinplot(x="Mutation", y="Mutation rate", hue="Type", data=all_data, hue_order=hue_order)
    g1.set_yscale("log")
    g1.set_ylim(10**-10, 10**-1)
    g1.set_xticklabels("")
    g1.set(xlabel="Major allele > Minor allele")
    pairs = [(("Mutation", "Simulated population sample"), ("Mutation", "Simulated population sample + sequence error"))]
    annot = Annotator(g1, pairs, x="Mutation", y="Mutation rate", hue="Type", data=all_data, hue_order=hue_order)
    annot.configure(test='Mann-Whitney', text_format='star', loc='outside', verbose=2,
                    comparisons_correction="Bonferroni")
    annot.apply_test()
    file_path = "sts.csv"
    with open(file_path, "w") as o:
        with contextlib.redirect_stdout(o):
            passage_g1, test_results = annot.annotate()

    g1.legend(bbox_to_anchor=(0.5, -0.3), loc="lower center", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("mutation_rate.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
