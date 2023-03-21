'''
Helper functions for loading and preparing data.
'''
from typing import List
from pathlib import Path
from dataclasses import dataclass, field

import pandas as pd
import numpy as np


def name_wrapper(
        filename: str, savepath: str = '', fileextension: str = '') -> Path:
    '''
    Prepends the filename with the path to save the file and the appropriate
    file extension.

    :param filename: Name of the file to store.
    :returns: Complete file name to store with path and file extension.
    '''
    return Path(savepath).joinpath(filename + fileextension)


@dataclass
class DataFileNames:
    '''
    Store names of files that are needed to visualize the results of the
    genetic algorithm.

    :param target: Name of the file containing the target data.
    :param grid_data: Name of the file containing the grid search data.
    :param single_obj_file: Name stem common to the files containing the logged
        data of the genetic algorithm runs with single objective optimization.
    :param multi_obj_file: Name stem common to the files containing the logged
        data of the genetic algorithm runs with multi objective optimization.
    '''
    target: str = 'attenuation_variations_averaged.pkl'
    grid_data: str = 'attenuation_grid_search.pkl'
    single_obj_file: str = 'ga_runs_length_const/ga_length_const_'
    multi_obj_file: str = 'ga_runs_length_and_amplitude/' \
        + 'ga_length_and_amplitude_'


@dataclass
class Data:
    """
    Store data needed to plot the results in
    :py:meth:`~plotting.PaperPlots.plot_genetic_algorithm_results`.
    """
    target: pd.DataFrame
    obs: pd.DataFrame  # observations extracted from grid data
    best_ind_single: List[pd.DataFrame]
    best_ind_multi: List[pd.DataFrame]
    cols_obs: pd.Index = field(init=False)

    def __post_init__(self):
        self.cols_obs = self.obs.columns[2:]

    def _get_obs_df(self, idx_drop: int) -> pd.DataFrame:
        '''
        Get data frame by dropping the `idx_drop` column from the `obs`
        data frame.

        :param idx_drop: Index expressing the column to be dropped.
        :returns: Data frame with dropped column such that it can be passed
            to :func:`the plot_heat_map`.
        '''
        return self.obs.drop(self.cols_obs[idx_drop], axis=1)

    def get_lambda_df(self) -> pd.DataFrame:
        '''
        Get data frame from obs that only contains the parameterization
        i.e. leak conductance and inter-compartment conductance and the
        length constant as columns.

        :returns: Data frame that can be passed directly to
            :func:`the plot_heat_map`.
        '''
        return self._get_obs_df(1)

    def get_amplitude_df(self) -> pd.DataFrame:
        '''
        Get data frame from obs that only contains the parameterization
        i.e. leak conductance and inter-compartment conductance and the
        amplitude in the first compartment as columns.

        :returns: Data frame that can be passed directly to
            :func:`the plot_heat_map`.
        '''
        return self._get_obs_df(0)

    def get_f2_df(self) -> pd.DataFrame:
        '''
        Get data frame from obs that only contains the parameterization
        i.e. leak conductance and inter-compartment conductance and the
        f2 fitness function as columns.

        :returns: Data frame that can be passed directly to
            :func:`the plot_heat_map`.
        '''
        fitness = np.linalg.norm(
            (self.obs[self.cols_obs].values - self.target)
            / self.target, axis=1)
        res = self.obs.copy()
        res = res.drop(self.cols_obs, axis=1)
        res["f2"] = fitness
        return res
