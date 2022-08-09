#! /powerapps/share/python-anaconda-2.7/bin/python
import pandas as pd
import numpy as np
import random

def choice(a, n, l):
    choices=[]
    if len(a)*l<n:
        raise ValueError("Impossible")
    s={k:n for k in a}
    for _ in range(n):
        r=random.choice(list(s))
        choices.append(r)
        s[r]-=1
        if s[r]==0:
            del(s[r])
    return choices


def analyse_ffpop(population_size, generations, chosen_pop, pos_no):
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
        sim_pop = np.ones(population_size, dtype=int)
        hiv_t_df["New_freq"] = 0
        for i in hiv_t_df.index:
            frequency = hiv_t_df["frequency"][i]
            np.random.seed(700)
            indices = np.random.choice(np.arange(sim_pop.size), replace=False, size=int(sim_pop.size * frequency))
            sim_pop[indices] = 0
            np.random.seed(800)
            new_array = np.random.choice(sim_pop, replace=False, size=chosen_pop)
            new_freq = 1-sum(new_array)/len(new_array)
            hiv_t_df["New_freq"][i] = new_freq
            # print(sim_pop[test], sim_pop[1])
        hiv_t_df.to_pickle("hiv.{}.pkl".format(str(g)))
        hiv_t_df.to_csv("hiv.{}.csv".format(str(g)))

def united_table_for_fits(generations):
    united_df = pd.DataFrame()
    columns = ["gen", "Base", "New_freq", "Pos"]
    for g in generations:
        hiv_df = pd.read_pickle("hiv.{0}.pkl".format(str(g)))
        united_df = pd.concat([united_df, hiv_df])
    united_df["Base"] = 1
    united_df = united_df[columns]
    united_df = united_df.rename(columns={"New_freq": "Freq", "gen": "Gen"})
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


def main():
    population_size = 1 * 10 ** 7
    generations = [2, 5, 8, 10, 12]
    chosen_pop = 10000
    pos_no = 100
    # analyse_ffpop(population_size, generations, chosen_pop, pos_no)
    df_all = united_table_for_fits(generations)
    df_all.to_csv("sim_data_hiv.txt", index=False, sep="\t")
    print(df_all)

if __name__ == '__main__':
    main()