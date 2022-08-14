#! /powerapps/share/python-anaconda-3.2019.7/bin/python3.7

"""
@Author: odedkushnir

"""

import os
import glob
import re

import pandas as pd
import numpy as np


def filter_rows_by_values(df, col, values):
    return df[~df[col].isin(values)]


def primerID_counter(file):
    table = pd.read_table(file)
    table = pd.DataFrame(table.columns.values[None, :], columns=table.columns).append(table).reset_index(drop=True)
    table.columns = range(table.shape[1])
    table = table.rename(columns={0: "Seq"})
    table = table.groupby(["Seq"])["Seq"].size().reset_index(name="Counts")
    pattern = re.compile("(----)", re.MULTILINE)
    table["Filter"] = np.where(table["Seq"].str.extract(pattern, expand=True) == "----", True, False)
    table = table[table["Filter"] == False]
    sum = table["Counts"].sum()
    mean = table["Counts"].mean()
    std = table["Counts"].std()
    return sum, mean, std


def main():
    file = "D:/My Drive/Studies/PhD/Projects/RV/RVB14/PrimerIDs/p2-2/primer_ids.txt"
    primerID_counter(file)


if __name__ == "__main__":
    main()