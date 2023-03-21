#!/usr/bin/env python3
'''
Attenuation experiment using genetic algorithms as optimizer.
'''
from typing import Callable, Dict, Sequence, Tuple, Union, Optional
from pathlib import Path

import random
import numpy as np
import pandas as pd

from deap import base, creator

from model_hw_mc_attenuation import Observation, fit_length_constant
from model_hw_mc_attenuation.base import Base as AttenuationExperiment
from model_hw_mc_attenuation.bss import default_conductance_limits
from model_hw_mc_attenuation.helper import extract_observation, \
    get_experiment, get_bounds, get_license_and_chip

from paramopt.genetic import tools
from paramopt.genetic.algorithms import algorithms


def get_evaluation_function(
        experiment: AttenuationExperiment,
        observation: Observation,
        target_obs: Union[float, np.ndarray],
        bounds: Optional[Tuple] = None) -> Callable:
    '''
    Define a function which extract the given observation from the given
    experiment.

    The returned function takes the parameterization of the experiment as an
    input and returns an observation.

    :param experiment: Experiment used to record the observation.
    :param observation: Type of observation to return by the function.
    :param target_obs: Target of the evaluation function.
    :param bounds: Bounds for exponential fit.
    :returns: Function which executes an experiment and returns the given
        fitness.
    :raises ValueError: If the provided observation is not supported.
    '''
    if observation == Observation.LENGTH_CONSTANT:
        def func_length_constant(params: np.ndarray) -> Tuple[float]:
            data = experiment.measure_response(np.asarray(params))
            obs = np.array([fit_length_constant(data[:, 0], bounds=bounds)])
            return (np.abs(target_obs - obs),)
        return func_length_constant

    if observation == Observation.LENGTH_AND_AMPLITUDE:
        def func_length_x_amplitude(
                params: np.ndarray) -> Tuple[float]:
            data = experiment.measure_response(np.asarray(params))
            obs = np.array([fit_length_constant(data[:, 0], bounds=bounds),
                            data[0, 0]])
            return (np.linalg.norm((obs - target_obs) / target_obs),)
        return func_length_x_amplitude

    raise ValueError(f'The observation "{observation}" is not supported.')


def _logbook2dataframe(logbook: tools.Logbook) -> pd.DataFrame:
    '''
    Create a :class:`pd.DataFrame` from a given logbook.

    The keys of the logbook will be translated to columns in the Dataframe.
    Values of the logbook which are sized objects must all have the same
    length. For each element in those sized objects a new row in the Dataframe
    is created.

    :param logbook: Logbook to create the DataFrame from.
    :returns: DataFrame with one row for each value in the logbook and one
        column for each key.
    :raises RuntimeError: If values of the logbook are sized objects and do not
        all have the same length.
    '''
    data = []
    # Each generation has an entry in the logbook
    for log_gen in logbook:
        # For some recorded data, we save a value for each individual in the
        # generation
        n_individuals = np.unique([len(v) for v in log_gen.values() if
                                   hasattr(v, '__len__')])
        if len(n_individuals) == 0:
            # Data is the same for all individuals
            data.append(list(log_gen.values()))
            continue
        if len(n_individuals) > 1:
            raise RuntimeError(
                f"All sized elements of the logbook must have the "
                f"same length. Found following lengths: {n_individuals}.")
        # Create one row for each individual
        for i in range(int(n_individuals)):
            data.append([v[i] if hasattr(v, '__len__') else v for v in
                         log_gen.values()])

    return pd.DataFrame(data, columns=logbook[0].keys())


# pylint:disable=too-many-locals
def main(target_data: pd.DataFrame,
         observation: Observation,
         algorithm: Callable, *,
         n_individuals: int,
         hyperparams: Dict,
         toolbox: base.Toolbox,
         global_parameters: bool = True) -> pd.DataFrame:
    '''
    Find parameters to replicate the given observation using genetic algorithms
    with the provided parameterization.

    This function supports various genetic algorithms with different
    evolutionary operators.
    The fitness is minimized in order to minimize the deviation from the
    supplied target of an observation which results from the evaluation of an
    individual.

    :param target_data: DataFrame from which the target is extracted.
    :param observation: Type of observation to extract from the target data.
    :param algorithm: Genetic algorithm to use.
    :param n_individuals: Number of individuals that will be generated on
        initialization of the algorithm.
    :param hyperparams: Hyperparameters supplied to the genetic algorithm.
    :param toolbox: Toolbox containing the evolutionary operators which are
        used in the genetic algorithm i.e. mate, select and mutate.
    :param global_parameters: Configure the leak and inter-compartment
        conductance globally, i.e. use the same values for all compartments.
    :returns: DataFrame containing all individuals created during the genetic
        algorithm and their corresponding fitness.
    '''
    attenuation_exp = get_experiment(target_data)
    limits = attenuation_exp.default_limits
    param_names = attenuation_exp.parameter_names(global_parameters)
    target_obs = extract_observation(target_data, observation).mean(0)

    if global_parameters:
        limits = limits[[0, -1]]

    toolbox.register("evaluate", get_evaluation_function(
        attenuation_exp, observation, target_obs,
        bounds=get_bounds(target_data)))

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)  # pylint:disable=no-member

    def create_individual(container: Sequence) -> Sequence:
        rng = np.random.default_rng()
        parameters = rng.integers(low=limits[:, 0].flatten(),
                                  high=limits[:, 1].flatten(),
                                  size=(1, np.shape(limits)[0]))
        return container(parameters.tolist()[0])

    toolbox.register("individual", create_individual, creator.Individual)  # pylint:disable=no-member
    toolbox.register("population", tools.initRepeat, list, toolbox.individual,  # pylint:disable=no-member
                     n=n_individuals)

    # Log used parameters and fitness
    stats = tools.Statistics(lambda ind: ind)

    def log_individuals(individuals, idx: int):
        result = []
        for ind in individuals:
            result.append(ind[idx])
        return result

    def log_fitness(individuals, idx: int = 0):
        fitnesses = []
        for ind in individuals:
            fitnesses.append(np.asarray(ind.fitness.values).flatten()[idx])
        return fitnesses

    for idx, param_name in enumerate(param_names):
        stats.register(param_name, log_individuals, idx=idx)
    stats.register("fitness", log_fitness)

    # Execute experiment
    _, log = algorithm(
        population=toolbox.population(),  # pylint:disable=no-member
        toolbox=toolbox,
        stats=stats,
        verbose=True,  # maybe set verbosity from loglevel?
        **hyperparams
    )

    # The columns of the logbook will generate the columns of the dataframe
    data = _logbook2dataframe(log)

    # Add information about the experiment parameterization
    data.attrs['limits'] = limits
    data.attrs['observation'] = observation.name
    data.attrs['algorithm'] = algorithm.__name__  # Get name of function used
    data.attrs['target'] = target_obs
    data.attrs['chip_id'] = get_license_and_chip()
    data.attrs['experiment'] = target_data.attrs['experiment']
    data.attrs['length'] = attenuation_exp.length
    data.attrs['calibration'] = attenuation_exp.calibration
    data.attrs['input_neurons'] = attenuation_exp.input_neurons
    data.attrs['input_weight'] = attenuation_exp.input_weight
    data.attrs['n_average'] = attenuation_exp.n_average
    data.attrs['n_individuals'] = n_individuals
    for key, value in hyperparams.items():
        data.attrs[key] = value

    def _get_evo_parameterization(data: pd.DataFrame) -> None:
        '''
        Append data's attrs with the used evoluationary operators and their
        parameterization that is derived from the toolbox.

        :param data: Dataframe to append attrs.
        '''
        evo_operators = ['mutate', 'mate', 'select']
        for evo_operator in evo_operators:
            data.attrs[evo_operator] = getattr(
                toolbox, evo_operator).func.__name__
            for key, value in getattr(toolbox, evo_operator).keywords.items():
                data.attrs[evo_operator + '_' + key] = value

    _get_evo_parameterization(data)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Perform a genetic algorithm to find appropriate "
                    "parameters to reproduce a given target observation. "
                    "As selection algorithm deap's selTrounament is used. ",
        conflict_handler='resolve')
    parser.add_argument("target",
                        help="Path to pickled DataFrame which contains "
                             "amplitudes of an attenuation experiment. The "
                             "mean over different runs will be used as a "
                             "target. Furthermore the calibration used to "
                             "generate the target data will be applied to the "
                             "experiment setup.",
                        type=str)
    parser.add_argument("-observation",
                        help="Determines what kind of observation is "
                             "extracted from the attenuation experiment and "
                             "the provided target",
                        type=str,
                        default=Observation.LENGTH_CONSTANT.name.lower(),
                        choices=[Observation.LENGTH_CONSTANT.name.lower(),
                                 Observation.LENGTH_AND_AMPLITUDE.name.lower()
                                 ])

    # Hyperparameters
    parser_hyperparams = parser.add_argument_group(
        'Genetic Algorithm - hyperparameters',
        description='Hyperparameters used for the genetic algorithm')
    parser_hyperparams.add_argument(
        "-ngen",
        help="Number of generations to evolve.",
        type=int,
        default=30)
    parser_hyperparams.add_argument(
        "-mutpb",
        help="Probability that a individual is mutated.",
        type=float,
        default=0.1)
    parser_hyperparams.add_argument(
        "-cxpb",
        help="Probability of mating two individuals.",
        type=float,
        default=0.5)

    parser.add_argument("-n_individuals",
                        help="Number of individuals per generation.",
                        type=int,
                        default=50)

    # Genetic algorithm & evolutionary operators
    parser.add_argument("-tournsize",
                        help="Tournament size of used for the tournament "
                             "selection.",
                        type=int,
                        default=3)
    parser.add_argument("-mutation",
                        help="Mutation operator to use.",
                        choices=['mutCustomBitFlip', 'mutUniformInt'],
                        type=str,
                        default='mutCustomBitFlip')
    parser.add_argument("-indpb",
                        help="Probability that a gene is mutated if individual"
                             " is picked for mutation.",
                        type=float,
                        default=0.5)
    parser.add_argument("-crossover",
                        help="Crossover operator to use.",
                        choices=['cxOnePoint', 'cxTwoPoint'],
                        type=str,
                        default='cxOnePoint')
    parser.add_argument("-algorithm",
                        help="Genetic algorithm to use.",
                        choices=algorithms.keys(),
                        type=str,
                        default='ea_elite')
    parser.add_argument("-n_elites",
                        help="Number of elites to pass directly to next "
                             "generation in the elitism genetic algorithm. "
                             "This is only relevant for the algorithm "
                             "ea_elite.",
                        type=int,
                        default=5)
    parser.add_argument("-mu",
                        help="Number of individuals to select for the next "
                             "generation. This is only relevant for the "
                             "algorithm eaMuPlusLambda.",
                        type=int)
    parser.add_argument("-lambda",
                        help="Number of children to produce in each "
                             "generation. This is only relevant for the "
                             "algorithm eaMuPlusLambda.",
                        dest='lambda_',
                        type=int)

    # Miscellaneous
    parser.add_argument("-seed",
                        help="Random seed for numpy and random.",
                        type=int)
    parser.add_argument('--global_parameters',
                        help="Use the same leak conductance in all "
                             "compartments and the same conductance between "
                             "compartments for all connections between "
                             "compartments.",
                        action='store_true')

    # Parse arguments
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    target_df = pd.read_pickle(args.target)

    # Create toolbox
    toolbox_ = base.Toolbox()

    toolbox_.register("mate", getattr(tools, args.crossover))
    toolbox_.register("select", tools.selTournament, tournsize=args.tournsize)
    toolbox_.register("mutate", getattr(tools, args.mutation),
                      indpb=args.indpb,
                      low=default_conductance_limits.T[0],
                      up=default_conductance_limits.T[1])

    algorithms_arguments = {'ea_elite': {'n_elites': args.n_elites},
                            'eaSimple': {},
                            'eaMuPlusLambda': {'mu': args.mu,
                                               'lambda': args.lambda_}}
    hyperparams_kwargs = {'ngen': args.ngen, 'mutpb': args.mutpb,
                          'cxpb': args.cxpb}
    hyperparams_kwargs.update(algorithms_arguments[args.algorithm])

    df = main(target_data=target_df,
              observation=Observation[args.observation.upper()],
              algorithm=algorithms[args.algorithm],
              n_individuals=args.n_individuals,
              hyperparams=hyperparams_kwargs,
              toolbox=toolbox_,
              global_parameters=args.global_parameters)
    df.attrs['target_file'] = str(Path(args.target).resolve())

    df.to_pickle('genetic_algorithm_evolution.pkl')
