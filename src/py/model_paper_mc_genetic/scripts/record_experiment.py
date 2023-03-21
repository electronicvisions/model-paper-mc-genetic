#!/usr/bin/env python3
"""
Summarize the experiments needed to aquire the data for the plots in the paper.

Run this script providing both the --grid_search and --genetic_algorithm flag
to record all data used in the paper.
The data used in the paper was recorded on hxcube9fpga0chip59 using the
calibration:
    /wang/data/calibration/hicann-dls-sr-hx/hxcube9fpga0chip59_1/stable/
    2022-10-09_1/spiking_cocolist.pbin

If a target is provided the experiments are run using the parameterization used
while creating the target data. This way the histogram data is not recorded.
"""
import pickle
from copy import deepcopy
from typing import Tuple, List, Optional
from pathlib import Path

import pandas as pd
import neo
from deap import base

from model_paper_mc_genetic.scripts.attenuation_ga import main as run_ga

from model_hw_mc_attenuation import Observation
from model_hw_mc_attenuation.helper import get_experiment
from model_hw_mc_attenuation.helper import grid_search as \
    run_grid_search
from model_hw_mc_attenuation.bss import _split_traces, \
    default_conductance_limits

from model_hw_mc_attenuation.scripts.record_variations import main as \
    record_variations
from model_hw_mc_attenuation.scripts.record_trace import main as record_trace

from paramopt.genetic import tools
from paramopt.genetic.algorithms import ea_elite


def data_histograms(calibration: Optional[str] = None,
                    save_path: Path = Path('')) -> None:
    '''
    Record the EPSP amplitudes for a given parameterization 1000 times.

    The recording is once performed without averaging and once with averaging
    over 10 trials.

    :param calibration: Calibration to use.
    :param save_path: Path to save results to.
    '''

    # Setup W69F0
    paper_params = {"length": 5,
                    "parameters": [511, 511],
                    "input_neurons": 5,
                    "input_weight": 63,
                    "repetitions": 1000,
                    "calibration": calibration}

    # Run without averaging
    data_single = record_variations(**paper_params, n_average=1)
    data_single.to_pickle(
        save_path.joinpath('attenuation_variations_single.pkl'))

    # Run with averaging and make sure same calibration is used
    if paper_params["calibration"] is None:
        paper_params["calibration"] = data_single.attrs["calibration"]
    data_averaged = record_variations(**paper_params, n_average=10)
    data_averaged.to_pickle(
        save_path.joinpath('attenuation_variations_averaged.pkl'))


def data_single_trace(target: pd.DataFrame, save_path: Path = Path("")
                      ) -> None:
    '''
    Record the membranes of each compartment of the chain during a synaptic
    input into the first compartment and save the traces.

    :param target: DataFrame from which the experiment parameterization can be
        extracted.
    :param save_path: Path to save results to.
    '''
    result = record_trace(length=target.attrs["length"],
                          input_neurons=target.attrs["input_neurons"],
                          input_weight=target.attrs["input_weight"],
                          n_average=1,
                          calibration=target.attrs["calibration"])
    neo.PickleIO(save_path.joinpath('membrane_traces.pkl')).write(result)


def data_sta_traces(target: pd.DataFrame, save_path: Path = Path("")) -> None:
    '''
    Record the membranes of each compartment of the chain during a synaptic
    input multiple times and save the traces.

    :param target: DataFrame from which the experiment parameterization can be
        extracted.
    :param save_path: Path to save results to.
    '''

    experiment = get_experiment(target)
    experiment.set_parameters(target.attrs["parameters"])
    traces = experiment._record_raw_traces()  # pylint:disable=protected-access
    splitted_traces = [_split_traces(trace, experiment.n_average) for trace in
                       traces]

    filename = 'attenuation_average_traces.pkl'
    with open(save_path.joinpath(filename), 'wb') as trace_file:
        pickle.dump(splitted_traces, trace_file)


def data_grid_search(target: pd.DataFrame, conductance: Tuple,
                     save_path: Path = Path("")) -> None:
    '''
    Execute a grid search using the provided conductance, execute the
    experiment at each setting and save the amplitudes.

    :param target: Target data from which the experiment configuration is
        derived.
    :param conductance: Tuple containing two lists each specifying the lower
        boundary, the uppper boundary and the step size of the grid for their
        respective parameter. The first list describes the parameter range of
        the leak conductance, the second the range of the inter-compartment
        conductance.
    :param save_path: Path to save results to.
    '''
    experiment = get_experiment(target)
    data = run_grid_search(
        experiment, g_leak=conductance[0], g_icc=conductance[1])
    data.attrs = target.attrs
    data.to_pickle(save_path.joinpath('attenuation_grid_search.pkl'))


def run_genetic_algorithm(target: pd.DataFrame, repetitions: int = 10,
                          save_path: Path = Path("")) -> None:
    '''
    Execute the genetic algorithm and save the data logged during execution.

    In total the genetic algorithm is executed 2 * `repetitions` times.
    For the first `repetitions` executions of the algorithm a single target
    optimization is done using the length constant as target provided by
    `target`.
    The subsequent `repetitions` executions are done with a multi-objective
    optimization considering both the length constant and the amplitude of the
    EPSP in the first compartment of the chain as targets.
    The results of the runs are stored in two subdirectories according to their
    optimization target.

    :param target: DataFrame with recorded amplitudes which will be used to
        extract the respective targets.
    :param repetitions: Number of repetitions of the genetic algorithm.
    :param save_path: Path to save results to. If in the provided path no
        directories with the names
        ga_runs_{length_const,length_const_and_amplitude} exist those
        directories will be created and the results will be saved in them.
    '''
    hyperparams = {'ngen': 30, 'mutpb': 0.1, 'cxpb': 0.5, 'n_elites': 5}
    n_individuals = 50

    toolbox = base.Toolbox()

    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mutate", tools.mutCustomBitFlip, indpb=0.5,
                     low=default_conductance_limits.T[0],
                     up=default_conductance_limits.T[1])

    for name, observation in [
            ('length_const', Observation.LENGTH_CONSTANT),
            ('length_and_amplitude', Observation.LENGTH_AND_AMPLITUDE)]:
        for i in range(repetitions):
            data = run_ga(target_data=target,
                          observation=observation,
                          algorithm=ea_elite,
                          n_individuals=n_individuals,
                          hyperparams=hyperparams,
                          toolbox=deepcopy(toolbox),
                          global_parameters=True)
            save_dir = save_path.joinpath(f'ga_runs_{name}')
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            data.to_pickle(save_dir.joinpath(f'ga_{name}_{i}.pkl'))


def main(save_path: Path, *,
         calibration: Optional[str] = None,
         grid_search: bool = False,
         conductance: Optional[List] = None,
         genetic_algorithm: bool = False,
         repetitions: int = 10) -> None:
    '''
    Run experiments presented in the paper.

    The results are saved to the directory `save_path`.
    A target is generated from repeating the chain attenuation experiment 1000
    times. A grid search is executed if `grid_search` is true using the step
    sizes defined in the `conductance` array.
    The genetic algorithm is executed `repetitions` times if
    `genetic_algorithm` is true.

    :param save_path: Path to which the results will be saved.
    :param calibration: Calibration file to use for target generation. If not
        provided the default calibration is used.
    :param grid_search: Boolean expressing whether a grid search is performed.
    :param conductance: List indicating the grid search parameter ranges that
        is interpreted like: [lower_bound, upper_bound, number_of_steps].
    :param genetic_algorithm: Boolean expressing whether the genetic algoritm
        is executed.
    :param repetitions: Number of times the genetic algorithm will be executed.
    '''
    # Record target and save it
    data_histograms(calibration, save_path)
    target_data = save_path.joinpath('attenuation_variations_averaged.pkl')

    # Record single trace and multiple traces
    data_single_trace(target_data, save_path)
    data_sta_traces(target_data, save_path)

    # Execute grid search
    if grid_search:
        if conductance is None:
            conductance = [10, 1020, 102]
        data_grid_search(target_data, (conductance, conductance), save_path)

    # Genetic algorithm experiments
    if genetic_algorithm:
        run_genetic_algorithm(target_data, repetitions, save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Perform the experiments to obtain the data necessary for "
                    "the paper.")
    parser.add_argument("-save_path",
                        help="Path to the directory where the data is "
                             "saved. For the genetic algorithm runs "
                             "subdirectories are created.",
                        type=str,
                        default='')
    parser.add_argument("-calibration",
                        type=str,
                        help="Path to portable binary calibration used for "
                             "target generation. If not provided the latest "
                             "nightly calibration is used.",
                        default=None)
    parser.add_argument("--grid_search",
                        help="If flag is provided the grid search is run.\n"
                             "Note: With the default resolution of 102 points "
                             "per parameter this might take O(4h).",
                        action="store_true")
    parser.add_argument("--genetic_algorithm",
                        help="If flag is provided the genetic algorithms are "
                             "run.\nNote: this might take O(12h) since one "
                             "run with 30 generations and 50 individuals "
                             "using the averging of the traces takes 40mins "
                             "and the algorithm is executed 20 times per "
                             "default.",
                        action="store_true")
    parser.add_argument("-repetitions",
                        type=int,
                        help="Number of times the genetic algorithm is "
                             "executed.",
                        default=10)

    args = parser.parse_args()

    main(save_path=Path(args.save_path),
         calibration=args.calibration,
         grid_search=args.grid_search,
         conductance=[10, 1020, 102],
         genetic_algorithm=args.genetic_algorithm,
         repetitions=args.repetitions)
