#! /powerapps/share/python-anaconda-2.7/bin/python
import pandas as pd
from FFPopSim import hivpopulation
import numpy as np


def run_ffpop():
    pop = hivpopulation(10000000)
    pop.set_replication_landscape()
    np.save("replicative_fitness_coefficients", pop.get_trait_additive())
    generations = [0, 2, 5, 8, 10, 12]

    for i in range(1, max(generations) + 1):
        pop.evolve()
        if i in generations:
            np.save("hiv." + str(i), pop.get_allele_frequencies())


def main():
    run_ffpop()


if __name__ == '__main__':
    main()