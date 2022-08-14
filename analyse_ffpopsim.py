#! /powerapps/share/python-anaconda-2.7/bin/python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import seaborn as sns
import scipy.stats as sts
from scipy.optimize import curve_fit
from sklearn.neighbors import KernelDensity


def error_rate_dist(data):
    data = data[data["Mutation"] == "U>C"]
    data["log_frequency"] = np.log10(data["Frequency"])
    mean = data["log_frequency"].mean()
    std = data["log_frequency"].std()
    return mean, std
    # data_set = data["log_frequency"].reset_index()
    # data_set = data_set.fillna(0)
    # data_set = data_set.drop(["index"], axis=1).reset_index()
    # data_set = data_set.rename(columns={"index": "count"})
    # model = sts.gaussian_kde(data_set["log_frequency"])
    # data_set["density"] = model.pdf(data_set["log_frequency"])
    # g = sns.lineplot(y="density", x="log_frequency", data=data_set)
    # plt.savefig("fig2.png")
    # g = sns.displot(x="log_frequency", data=data_set, kind="kde")
    # plt.savefig("fig1.png")


# def seq_error_noise(pop_arr, mu, sigma):
#     np.random.seed(700)
#     for i in pop_arr:
#         rand_number = random.uniform(0, 1)
#         error_number = np.exp(random.gauss(mu, sigma))
#         if rand_number < error_number:
#             pop_arr[i] = 1-pop_arr[i]
#     return pop_arr


def seq_error_noise(pop_arr, error_rate_data):
    error_rate_data = error_rate_data[error_rate_data["Mutation"] == "U>C"]
    np.random.seed(700)
    for i in pop_arr:
        rand_number = random.uniform(0, 1)
        row = error_rate_data.sample()
        frequency = row["Frequency"].values[0]
        if rand_number < frequency:
            pop_arr[i] = 1-pop_arr[i]
    return pop_arr


def analyse_ffpop(population_size, generations, chosen_pop, pos_no, data_error_rate, mu, sigma):
    replicative = np.load("replicative_fitness_coefficients.npy")
    replicative_trans = np.array(([replicative]))
    replicative_trans = replicative_trans.T
    replicative_df = pd.DataFrame(replicative_trans, columns=["coefficient"])
    replicative_df.reset_index(inplace=True)
    replicative_df = replicative_df.rename(columns={'index': 'Pos'})
    replicative_df["Pos"] = replicative_df["Pos"].astype(float)
    replicative_df["fitness"] = np.exp(replicative_df["coefficient"])
    replicative_df["fitness"] = replicative_df["fitness"].apply(lambda x: float(x))
    replicative_df["mutation"] = np.where(replicative_df["fitness"] == 1, "Synonymous", "else")
    replicative_df["Filter"] = np.where(replicative_df["fitness"] == 1, True, False)
    replicative_df_syn = replicative_df[replicative_df["Filter"] == True]
    replicative_df_syn.reset_index(inplace=True)
    replicative_df_syn = replicative_df_syn.drop(columns="index")
    replicative_df_syn["Pos"] = replicative_df_syn["Pos"].astype(int)
    replicative_df_100 = replicative_df_syn.loc[replicative_df_syn.index < pos_no]
    positions_series = replicative_df_100["Pos"]

    for g in generations:
        hiv = np.load("hiv.{0}.npy".format(str(g)))
        hiv_t = np.array([hiv])
        hiv_t = hiv_t.T
        hiv_t_df = pd.DataFrame(hiv_t, columns=["frequency"])
        hiv_t_df.reset_index(inplace=True)
        hiv_t_df = hiv_t_df.rename(columns={'index': 'Pos'})
        hiv_t_df = hiv_t_df.merge(positions_series, how="inner")
        hiv_t_df["Pos"] = hiv_t_df["Pos"].astype(int)
        hiv_t_df["gen"] = g
        sim_pop = np.zeros(population_size, dtype=int)
        hiv_t_df["New_freq"] = 0
        hiv_t_df["Noisy_freq"] = 0
        for i in hiv_t_df.index:
            frequency = hiv_t_df["frequency"][i]
            np.random.seed(700)
            indices = np.random.choice(np.arange(sim_pop.size), replace=False, size=int(sim_pop.size * frequency))
            sim_pop[indices] = 1
            np.random.seed(800)
            new_array = np.random.choice(sim_pop, replace=False, size=chosen_pop)
            new_freq = sum(new_array)/len(new_array)
            hiv_t_df["New_freq"][i] = new_freq
            sim_pop_noisy = seq_error_noise(new_array, mu, sigma)
            noisy_freq = sum(sim_pop_noisy)/len(sim_pop_noisy)
            hiv_t_df["Noisy_freq"][i] = noisy_freq
        hiv_t_df.to_pickle("hiv.{}.pkl".format(str(g)))
        hiv_t_df.to_csv("hiv.{}.csv".format(str(g)))
    return new_array, sim_pop_noisy


def united_table_for_fits(generations, freq_type="New_freq"):
    united_df = pd.DataFrame()
    columns = ["gen", "Base", freq_type, "Pos"]
    for g in generations:
        hiv_df = pd.read_pickle("hiv.{0}.pkl".format(str(g)))
        united_df = pd.concat([united_df, hiv_df])
    united_df["Base"] = 0
    united_df = united_df[columns]
    united_df = united_df.rename(columns={freq_type: "Freq", "gen": "Gen"})
    united_df.sort_values(by=['Pos'])
    grouped = united_df.groupby(['Pos', 'Gen'])
    df_all = pd.DataFrame(columns=["Gen", "Base", "Freq", "Pos"])
    for group in grouped:
        df = pd.concat([group[1]]*2, ignore_index=True)
        df.at[0, 'Freq'] = 1 - df.at[1, 'Freq']
        df.at[0, 'Base'] = 0
        df.at[1, 'Base'] = 1
        df_all = pd.concat([df_all, df], ignore_index=True)
    df_all["Pos"] = df_all["Pos"].astype(int)
    return df_all


def plot_sim_noisy(generations):
    x_ticks = ["0", "", "2", "", "", "5", "", "", "8", "", "10", "", "12", ""]
    x_order = range(0, 14, 1)
    united_df = pd.DataFrame()
    for g in generations:
        hiv_df = pd.read_pickle("hiv.{0}.pkl".format(str(g)))
        united_df = pd.concat([united_df, hiv_df])
    united_df_sim = united_df.copy()
    united_df_sim = united_df_sim.drop(["New_freq", "Noisy_freq"], axis=1)
    united_df_sim["Type"] = "HIV Simulated population"
    united_df_sample = united_df.copy()
    united_df_sample = united_df_sample.drop(["frequency", "Noisy_freq"], axis=1)
    united_df_sample["Type"] = "HIV Simulated sample"
    united_df_sample = united_df_sample.rename(columns={"New_freq": "frequency"})
    united_df_noisy = united_df.copy()
    united_df_noisy = united_df_noisy.drop(["frequency", "New_freq"], axis=1)
    united_df_noisy["Type"] = "HIV Simulated sample + Sequence Error"
    united_df_noisy = united_df_noisy.rename(columns={"Noisy_freq": "frequency"})
    rv_data = pd.read_csv("table.csv")
    rv_data = rv_data[["Pos", "Frequency", "Type", "passage"]]
    rv_data = rv_data[rv_data["Type"] == "Synonymous"]
    rv_data = rv_data.drop(["Type"], axis=1)
    rv_data = rv_data.rename(columns={"Frequency": "frequency", "passage": "gen", "Type": "Mutation Type"})
    rv_data["Type"] = "RVB14 RdRp Synonymous mutations"
    data = pd.concat([united_df_sim, united_df_sample, united_df_noisy, rv_data])
    plot = sns.catplot(x="gen", y="frequency", data=data, hue="Type", kind="box", order=x_order)
    plot.set(xlabel="Passage", ylabel="Variant Frequency", yscale="log", ylim=(10 ** -5, 10 ** -2), xticklabels=x_ticks)
    plt.savefig("fig4.png")
    plt.close()


def main():
    population_size = 1 * 10 ** 7
    generations = [2, 5, 8, 10, 12]
    chosen_pop = 10000
    pos_no = 100
    data_error_dist = pd.read_csv("error_table.csv")
    # mu, sigma = error_rate_dist(data_error_dist)
    # sim_pop_new, pop_arr_noise = analyse_ffpop(population_size, generations, chosen_pop, pos_no, data_error_dist)
    # np.save("sim_pop_new.npy", sim_pop_new)
    # np.save("pop_arr_noise.npy", pop_arr_noise)

    # df_all = united_table_for_fits(generations, freq_type="Noisy_freq")
    # df_all.to_csv("sim_data_hiv_noisy.txt", index=False, sep="/t")

    plot_sim_noisy(generations)

    # sim_pop_new = np.load("sim_pop_new.npy")
    # pop_arr_noise = seq_error_noise(sim_pop_new, data_error_dist)

    # pop_arr_noise = np.load("pop_arr_noise.npy")
    # print(pop_arr_noise)

    # nums = []
    # for i in range(10000):
    #     temp = random.gauss(mu, sigma)
    #     nums.append(temp)
    #
    # # plotting a graph
    # plt.hist(nums, bins=200)
    # plt.savefig("Fig3.png")


if __name__ == '__main__':
    main()