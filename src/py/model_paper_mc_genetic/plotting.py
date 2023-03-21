'''
Helper classes and functions used for plotting.
'''
from typing import Callable, Dict, Optional, List, Tuple
from functools import partial
import pickle

import numpy as np
import pandas as pd
import neo
import quantities as pq

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from model_paper_mc_genetic.helpers import name_wrapper, DataFileNames, Data

from model_hw_mc_attenuation import Observation, extract_psp_heights, \
    exponential_decay, fit_exponential
from model_hw_mc_attenuation.helper import extract_observation
from model_hw_mc_attenuation.bss import _average_traces

from model_hw_mc_attenuation.plotting.grid_search import plot_heat_map, \
    create_obs_dataframe, plot_contour_lines


def set_latex_style() -> None:
    """
    Adapt matplotlib's runtime configuration such that it uses latex style with
    custom parameterization.
    """
    matplotlib.use("pgf")
    # Setup matplotlib to use latex for output
    pgf_with_latex = {
        "pgf.texsystem": "pdflatex",
        "text.usetex": True,
        "font.family": "serif",
        # Blank entries should lead to inheriting the fonts from the document
        "font.serif": [],
        "font.sans-serif": [],
        "font.monospace": [],
        # Specify sizes of text
        "axes.labelsize": 9,
        "font.size": 8,
        "legend.fontsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "pgf.preamble": "\n".join([
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[detect-all,locale=DE]{siunitx}",
        ])
    }
    matplotlib.rcParams.update(pgf_with_latex)


class PaperPlots:
    '''
    Class containing functions which can generate the plots used in the paper.
    '''
    def __init__(self, figure_width: float, figure_height,
                 data_wrapper: Optional[Callable] = None,
                 save_wrapper: Optional[Callable] = None):
        self.figure_width = figure_width
        self.figure_height = figure_height
        self.data_wrapper = data_wrapper
        self.save_wrapper = save_wrapper

        if self.data_wrapper is None:
            self.data_wrapper = partial(name_wrapper, '', '')
        if self.save_wrapper is None:
            self.save_wrapper = partial(name_wrapper, '', '.png')

    def plot_single_trace_attenuation(
            self, filename: str = "membrane_traces.pkl") -> None:
        '''
        Visualize the EPSP propagating along the compartment chain.

        :param filename: Name of the file which contains the traces.
        '''
        # Load data
        reader = neo.PickleIO(filename=self.data_wrapper(filename))
        block = reader.read_block()

        # Prepare data
        def _prepare_trace(
                trace: neo.IrregularlySampledSignal,
                ref_time: pq.ms) -> neo.IrregularlySampledSignal:
            '''
            Subtract baseline from trace and cut it in time relative to the
            provided `ref_time`.

            :param trace: Trace to be prepared.
            :param ref_time: Reference time for the time slice.
            :returns: Modified signal.
            '''

            time_slice = (ref_time * 0.9, ref_time * 2) * pq.ms
            sliced_trace = trace.time_slice(*time_slice)
            baseline = np.mean(sliced_trace.magnitude[:100])

            return neo.IrregularlySampledSignal(
                sliced_trace.times,
                (sliced_trace.magnitude - baseline) * trace.units)

        def _norm_traces(traces: List[neo.IrregularlySampledSignal]
                         ) -> List[neo.IrregularlySampledSignal]:
            '''
            Norm all signals to the maximal amplitude of all signals.

            :param traces: List of signals.
            :returns: List of normed signals.
            '''
            norm_fac = np.max([trace.max() for trace in traces])
            return [trace / norm_fac for trace in traces]

        spike_time = block.annotations["spike_times"][0]
        traces = _norm_traces(
            [_prepare_trace(trace, spike_time) for trace in
             block.segments[0].irregularlysampledsignals])

        # Plot data
        fig, axes = plt.subplots(
            1, 1, figsize=(self.figure_width / 2, self.figure_height), dpi=200)

        skips = 3
        for n_comp, trace in enumerate(traces):
            axes.plot(trace.times[::skips], trace.magnitude[::skips],
                      c=f'C{n_comp}', alpha=0.8, rasterized=True)

        if 'siunitx' in matplotlib.rcParams['pgf.preamble']:
            axes.set_xlabel(r'time [$\si{\micro\second}$]' + '\n')
        else:
            axes.set_xlabel(r'time [$\mu$s]' + '\n ')
        axes.set_ylabel(r'$U_\mathrm{m}$ [a.U.]')

        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

        fig.tight_layout(pad=0)
        fig.savefig(self.save_wrapper("epsp_in_different_compartments"))

    def plot_single_trace_exp_decay(
            self, filename: str = "membrane_traces.pkl") -> None:
        '''
        Visualize the decay of the EPSP amplitudes along the compartment chain
        and fit an exponential to it.

        The amplitudes are visualized in a scatter plot and the fit function
        which is used to determine the length constant is used for the
        exponential fit.

        :param filename: Name of the file which contains the traces from which
            the amplitudes are extracted.
        '''
        # Load data
        reader = neo.PickleIO(filename=self.data_wrapper(filename))
        block = reader.read_block()

        # Prepare data
        heights = extract_psp_heights(
            block.segments[0].irregularlysampledsignals)[:, 0]
        popt = fit_exponential(heights)

        # Plot
        fig, axes = plt.subplots(
            1, 1, figsize=(self.figure_width / 2, self.figure_height), dpi=200)

        comp = np.arange(0, len(heights) - 1, 0.01)
        axes.plot(comp, exponential_decay(comp, *popt) / heights[0], c='gray')

        for n_comp, height in enumerate(heights):
            axes.scatter(
                [n_comp], [height] / heights[0], c=f'C{n_comp}', zorder=10)

        # Set only integer ticks
        axes.xaxis.set_major_locator(MaxNLocator(integer=True))

        axes.set_ylabel(r'$U_\mathrm{m, max}$ [a.U.]')
        axes.set_xlabel('location\n[compartment]')
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

        fig.tight_layout(pad=0)
        fig.savefig(self.save_wrapper("attenuation_of_EPSP"))

    def plot_histograms(
            self, filename_single: str = "attenuation_variations_single.pkl",
            filename_average: str = "attenuation_variations_averaged.pkl"
    ) -> None:
        '''
        Visualize the trial-to-trial variations of the observables using single
        execution and STA.

        As observables the length constant and the EPSP amplitude in the first
        are considered.

        :param filename_single: Name of file containing the data from single
            recording.
        :param filename_average: Name of file containing the data using STA for
            recording.
        '''

        # Load data
        data_single = pd.read_pickle(self.data_wrapper(filename_single))
        data_average = pd.read_pickle(self.data_wrapper(filename_average))

        # Prepare data
        lambda_single = extract_observation(
            data_single, Observation.LENGTH_CONSTANT)
        lambda_average = extract_observation(
            data_average, Observation.LENGTH_CONSTANT)

        amplitude_single = extract_observation(
            data_single, Observation.AMPLITUDE_00)
        amplitude_average = extract_observation(
            data_average, Observation.AMPLITUDE_00)

        # Plot
        fig, axes = plt.subplots(
            1, 2, figsize=(self.figure_width, self.figure_height), sharey=True,
            dpi=250)

        bins = axes[0].hist(lambda_single, bins=20, alpha=0.5)[1]
        axes[0].hist(lambda_average, bins=bins, alpha=0.5)

        bins = axes[1].hist(amplitude_single, bins=20, alpha=0.5)[1]
        axes[1].hist(amplitude_average, bins=bins, alpha=0.5)

        # Legends
        labels = [r"$\bar{\lambda}_\mathrm{emp}= $"
                  + rf"{data.mean():.2f}$\pm$" + rf"{data.std():.2f}"
                  for data in [lambda_single, lambda_average]]
        axes[0].legend(labels=labels, fontsize=6, loc='upper right')
        labels = [rf"$\bar{{h}}^0= ${data.mean():.0f}$\pm${data.std():.0f}"
                  for data in [amplitude_single, amplitude_average]]
        axes[1].legend(labels=labels, fontsize=6)

        # Labels
        axes[0].set_xlabel(r'length constant $\lambda_\mathrm{emp}$'
                           + '\n' + r'[$\mathrm{compartments}$]')
        axes[1].set_xlabel('amplitude $h^0$\n[MADC LSB]')
        axes[0].set_ylabel('count')

        fig.tight_layout(pad=0)
        fig.savefig(self.save_wrapper("hist_trial_to_trial"))

    def plot_sta(
            self, filename: str = "attenuation_average_traces.pkl") -> None:
        """
        Visualize the spike-triggered average (STA) process used in the paper.

        Each trace used to calculate the STA are plotted in grey and the
        resulting STA trace is plotted above in color corresponding to the
        compartment. The baseline is subtracted from each trace such that one
        can see the amplitude better.

        :param filename: Name of file containing traces to plot.
        """
        # Load data
        with open(self.data_wrapper(filename), "rb") as data_file:
            data = pickle.load(data_file)

        # Prepare data
        sta_traces = [_average_traces(traces_comp) for traces_comp in data]

        # Plot data
        fig, axes = plt.subplots(
            1, 1, figsize=(self.figure_width, self.figure_height), dpi=200)

        # Iterate over compartments
        for n_comp, comp_data in enumerate(data):
            # Use first spike as time reference
            spike_time = comp_data[0].annotations["input_spikes"][0]
            times4slice = (spike_time, spike_time * 1.8) * pq.ms

            # Plot STA traces
            sta_baseline = np.mean(sta_traces[n_comp].magnitude[:200])
            sta_trace = sta_traces[n_comp].time_slice(*times4slice)
            sta_trace.times = sta_trace.times - spike_time
            axes.plot(sta_trace.times.rescale(pq.us),
                      sta_trace.magnitude - sta_baseline,
                      c=f"C{n_comp}", rasterized=True, zorder=1000)

            # Plot every trace from which the STA was calculated
            for trace in comp_data:
                # Adjust times
                baseline = np.mean(trace.magnitude[:200])
                trace.times = sta_traces[n_comp].times
                trace = trace.time_slice(*times4slice)
                axes.plot(trace.times.rescale(pq.us) - spike_time,
                          trace.magnitude - baseline, c='gray', alpha=0.5,
                          rasterized=True)

        # Layout of plot
        axes.spines['right'].set_visible(False)
        axes.spines['top'].set_visible(False)

        if 'siunitx' in matplotlib.rcParams['pgf.preamble']:
            axes.set_xlabel(r'time [$\si{\micro\second}$]')
        else:
            axes.set_xlabel(r'time [$\mu$s]')
        axes.set_ylabel("Baseline subtracted\nMembrane potential\n"
                        + r"$U_\mathrm{m}$ [MADC LSB]")

        fig.tight_layout(pad=0)
        fig.savefig(self.save_wrapper("STA_vs_traces"))

    def plot_grid_search(
            self, filename: str = 'attenuation_grid_search.pkl') -> None:
        '''
        Visualize the space of the observables in dependence of the experiment
        parameterization.

        Two subplots are generated. One plot is visualizing the dependency of
        the length constant on the leak conductance and inter-compartment
        conductance. The other plot shows the dependency of the EPSP amplitude
        in the first compartment of the chain on the leak conductance and the
        inter-compartment conductance.

        :param filename: Name of the file containing the grid search data.
        '''
        # Load data
        data = pd.read_pickle(self.data_wrapper(filename))

        # Prepare data
        df_lambdas = create_obs_dataframe(data, Observation.LENGTH_CONSTANT)
        df_amplitudes = create_obs_dataframe(data, Observation.AMPLITUDE_00)

        # Plot
        height = ((self.figure_width) * 1.3) / 2
        fig, axes = plt.subplots(1, 2, figsize=(self.figure_width, height),
                                 sharey=True, dpi=200)

        # Mesh
        # _vis_data_in_grid(df_lambdas, fig, axes[0],
        mesh_lambdas = plot_heat_map(axes[0], df_lambdas)
        mesh_amplitudes = plot_heat_map(axes[1], df_amplitudes)
        mesh_lambdas.set_cmap('viridis')
        mesh_lambdas.set_rasterized(True)
        mesh_amplitudes.set_cmap('viridis')
        mesh_amplitudes.set_rasterized(True)

        # Contours
        lvl_lambda = [0.5, 0.8, 1.1, 1.4, 1.7]
        _add_contours(axes[0], df_lambdas, lvl_lambda)

        lvl_amp = [150, 180, 210, 240, 270]
        _add_contours(axes[1], df_amplitudes, lvl_amp)

        # Labels
        axes[0].set_xlabel(r"$\mathrm{bias}\;g_\mathrm{l}$ [LSB]")
        axes[1].set_xlabel(r"$\mathrm{bias}\;g_\mathrm{l}$ [LSB]")
        axes[0].set_ylabel(r"$\mathrm{bias}\;g_\mathrm{ic}$ [LSB]")
        axes[1].set_ylabel(r"")

        # Add colorbars for both panes
        cb_length = _add_colorbar_beneath(fig, axes[0], mesh_lambdas)
        cb_length.set_label(r'length constant $\lambda_\mathrm{emp}$'
                            + '\n[compartments]')

        cb_height = _add_colorbar_beneath(fig, axes[1], mesh_amplitudes)
        cb_height.set_label(r'amplitude $h^0$' + '\n[MADC LSB]')

        fig.tight_layout(pad=0)
        plt.savefig(self.save_wrapper("grid_search_length_height"))

    def _loop_data(
            self, base_name: str, n_files: int = 10) -> List[pd.DataFrame]:
        '''
        Load DataFrames with common name stem and only different index ending
        and return them in a list.

        The files should have following naming scheme: nameX. Where X is an
        integer.

        :param base_name: Name of the file stem common between the files that
            should be loaded.
        :param n_files: Specifies the range of integers which are extended to
            the common name stem.
        :returns: List of DataFrames.
        '''
        return [pd.read_pickle(self.data_wrapper(base_name + str(i) + '.pkl'))
                for i in range(n_files)]

    def plot_genetic_algorithm_performance(  # pylint:disable=invalid-name
            self,
            filename: str = 'ga_runs_length_const/ga_length_const_',
            fitness: str = 'fitness',
            filename_target: str = 'attenuation_variations_averaged.pkl'
    ) -> None:
        '''
        Visualize the average and best performance of the evolving genetic
        algorithm of multiple executions.

        Generates two subplots. One illustrating the generations average
        performance over multiple executions and the other showing the minimum
        (best) performance. Additionally, the expected standard deviation of
        the observable when executing the experiment is displayed by a dashed
        horizontal line. The data for that is extracted from the file with name
        `filename_target`.

        :param filename: Name stem common to the files containing the logged
            data of the genetic algorithm runs.
        :param fitness: Name of the DataFrame's column for which the
            generation's average and minimum is visualised.
        :param filename_target: Name of the file where the target data is
            stored.
        '''

        # Load data
        data = self._loop_data(filename, 10)
        target = pd.read_pickle(self.data_wrapper(filename_target))

        # Prepare data
        means = [df.groupby(['gen']).mean()[fitness] for df in data]
        bests = [df.groupby(['gen']).min()[fitness] for df in data]
        target_lambda = extract_observation(
            target, Observation.LENGTH_CONSTANT)

        # Plot
        fig, axes = plt.subplots(
            1, 2, figsize=(self.figure_width, self.figure_height))

        for ax in axes:
            ax.set_yscale('log')
            ax.axhline(np.std(target_lambda), ls=':', color='black')
            ax.grid()

        for mean in means:
            axes[0].plot(mean.index, mean.values, c='grey', alpha=0.8)
        for best in bests:
            axes[1].plot(best.index, best.values, c='grey', alpha=0.8)

        # Labels
        axes[0].set_ylabel(
            r'fitness $|\hat{\lambda}_\mathrm{emp} - '
            + r'\lambda_\mathrm{emp}|$' + '\n[compartments]')
        for ax in axes:
            ax.set_xlabel('generation')
        axes[0].set_title("Population average")
        axes[1].set_title("Best individual")

        fig.tight_layout(pad=0)
        plt.savefig(self.save_wrapper("performance_ga"))

    @staticmethod
    def _get_best_individual(data: pd.DataFrame, fitness: str) -> pd.DataFrame:
        """
        Get row of DataFrame with minimal `fitness` of the last generation.

        :param data: Data frame containing logbook of genetic algorithm run.
        :param fitness: Column name for which minimum value is searched for.
        :returns: Row of DataFrame with minimal `fitness`.
        """
        last_gen = data.loc[(data['gen'] == data['gen'].max())]
        return last_gen.loc[last_gen[fitness].idxmin()]

    def _get_data(self, files: DataFileNames, fitness: str, n_ga_runs: int
                  ) -> Data:
        '''
        Read in all data, prepare it and return the prepared data in as
        a :class:`Data` object.

        :param files: Data class storing the name of the files which contain
            the data to plot the results.
        :param fitness: Name of the column used for the fitness as saved by the
            genetic algorithm.
        :param n_ga_runs: Number of files with common name stem i.e. number of
            runs of the genetic algorithm.
        :returns: Data needed to plot results.
        '''
        target = pd.read_pickle(self.data_wrapper(files.target))
        grid_data = pd.read_pickle(self.data_wrapper(files.grid_data))

        single_obj = self._loop_data(files.single_obj_file, n_ga_runs)
        multi_obj = self._loop_data(files.multi_obj_file, n_ga_runs)

        # Prepare data
        observation = Observation.LENGTH_AND_AMPLITUDE

        target = extract_observation(
            target, observation=observation).mean(0)
        obs = create_obs_dataframe(grid_data, Observation.LENGTH_CONSTANT)
        tmp = create_obs_dataframe(grid_data, Observation.AMPLITUDE_00)
        amp = tmp[tmp.columns[-1]]
        obs = obs.join(amp)

        single_obj_row = [self._get_best_individual(df, fitness) for df in
                          single_obj]
        multi_obj_row = [self._get_best_individual(df, fitness) for df in
                         multi_obj]
        return Data(target, obs, single_obj_row, multi_obj_row)

    @staticmethod
    def _vis_data_in_grid(data, fig, axes, lvl, cb_label: str = ''):
        mesh_data = plot_heat_map(axes, data)
        mesh_data.set_cmap('viridis')
        mesh_data.set_rasterized(True)

        _add_contours(axes, data, levels=lvl)
        color_bar = _add_colorbar_beneath(fig, axes, mesh_data)
        color_bar.set_label(cb_label)

    @staticmethod
    def _mark_solution(axes: matplotlib.axes.Axes, data: pd.DataFrame,
                       marker_kwargs: Optional[Dict] = None) -> None:
        '''
        Mark the parameterization stored in data in the provided axes as
        scatter plot.

        :param axes: Axes to plot scatter into.
        :param data: Parameterization i.e. leak conductance and
            inter-compartment conductance.
        :marker_kwargs: Any kwargs that should be passed to the scatter plot.
        '''
        axes.scatter(data['g_leak'], data['g_icc'], **marker_kwargs)

    def _mark_solutions(self,
                        axes: matplotlib.axes.Axes,
                        solutions: List[pd.DataFrame],
                        marker_kwargs: Dict) -> None:
        for solution in solutions:
            self._mark_solution(axes, solution, marker_kwargs)

    @staticmethod
    def _get_target_df(filename: str = "attenuation_variations_single.pkl"
                       ) -> pd.DataFrame:
        data = pd.read_pickle(filename)
        g_leak = data.attrs['parameters'][0]
        g_icc = data.attrs['parameters'][data.attrs['length']]
        return pd.DataFrame([[g_leak, g_icc]], columns=['g_leak', 'g_icc'])

    def plot_genetic_algorithm_results(
            self,
            files: DataFileNames = DataFileNames(),
            fitness: str = 'fitness',
            n_ga_runs: int = 10) -> None:
        '''
        Visualize the resulting best individuals of the last generation in the
        observable/solution space.

        Two subplots are generated. The first subplot shows the length constant
        in dependence of the leak conductance and the inter-compartment
        conductance.
        The second plot shows the fitness in dependency of the leak and
        inter-compartment conductance. In both subplots the best individuals
        from multiple genetic algorithm executions with different fitness
        functions are shown as scatter plots. The different fitness functions
        are indicated by the color of the markers.
        In the subplot of the length constant the target length constant is
        highlighted by a red contour line as well as the target amplitude.

        :param files: Data class storing the name of the files which contain
            the data to plot the results.
        :param fitness: Name of the column used for the fitness as saved by the
            genetic algorithm.
        :param n_ga_runs: Number of files with common name stem i.e. number of
            runs of the genetic algorithm.
        '''
        data = self._get_data(files, fitness, n_ga_runs)

        height = ((self.figure_width) * 1.3) / 2
        fig, axes = plt.subplots(1, 2, figsize=(self.figure_width, height),
                                 sharey=True, dpi=200)

        # Plot heat map with contours for lambda and fitness f2
        self._vis_data_in_grid(
            data.get_lambda_df(), fig, axes[0], lvl=[0.5, 0.8, 1.4, 1.7],
            cb_label=r'length constant $\lambda_\mathrm{emp}$'
                     + '\n[compartments]')

        self._vis_data_in_grid(
            data.get_f2_df(), fig, axes[1], lvl=[0.1, 0.2, 0.4, 0.6],
            cb_label=r'fitness $f_2$')

        # Contours for targets
        target_lam = np.round(data.target[0], 2)
        target_amp = int(np.round(data.target[1], 0))
        _add_contours(axes[0], data.get_lambda_df(), levels=[target_lam],
                      kwargs={'colors': 'red'}, label_loc=[(350, 450)])
        _add_contours(
            axes[0], data.get_amplitude_df(), levels=[target_amp],
            kwargs={'colors': 'red', 'linestyles': ':'})

        # Mark parameters of the best individuals of the genetic algorithm runs
        # in the solution space with scatter plots
        marker_kwargs = {'marker': 'X', 'zorder': 1000, 'linewidth': 0.5,
                         'edgecolor': 'k', 's': 30, 'c': 'C0', 'alpha': 0.8}
        self._mark_solutions(axes[0], data.best_ind_single, marker_kwargs)
        self._mark_solutions(axes[1], data.best_ind_single, marker_kwargs)
        marker_kwargs.update({'c': 'C4'})
        self._mark_solutions(axes[0], data.best_ind_multi, marker_kwargs)
        self._mark_solutions(axes[1], data.best_ind_multi, marker_kwargs)

        # Mark parameter used for target generation
        marker_kwargs.update({'zorder': 10000, 'c': 'C3'})
        for axs in axes:
            self._mark_solution(axs, self._get_target_df(), marker_kwargs)

        # Labels
        axes[0].set_xlabel(r"$\mathrm{bias}\;g_\mathrm{l}$ [LSB]")
        axes[1].set_xlabel(r"$\mathrm{bias}\;g_\mathrm{l}$ [LSB]")
        axes[0].set_ylabel(r"$\mathrm{bias}\;g_\mathrm{ic}$ [LSB]")
        axes[1].set_ylabel(r"")

        fig.tight_layout(pad=0)
        fig.savefig(self.save_wrapper("grid_search_length_multiobj"))


def _add_colorbar_beneath(
        fig: matplotlib.figure.Figure,
        axes: matplotlib.axes.Axes,
        data: matplotlib.collections.QuadMesh) -> matplotlib.colorbar.Colorbar:
    '''
    Creates a new axis beneath the provided `axes` and puts a color bar
    there using the range provided by `data`.

    :param fig: Figure object where axes is located in.
    :param axes: Axis where the color bar is plotted beneath.
    :param data: Value range the color bar will describe.
    :returns: The generated color bar object.
    '''
    divider = make_axes_locatable(axes)
    cax = divider.append_axes("bottom", size="5%", pad=0.50)
    return fig.colorbar(data, cax=cax, orientation='horizontal')


def _add_contours(
        axes: matplotlib.axes.Axes, data: pd.DataFrame, levels: List,
        kwargs: Optional[Dict] = None, label_loc: Optional[List[Tuple]] = None
) -> None:
    '''
    Adds contour plot to given axis and labels them.

    The underlying data is smoothed with a Gaussian kernel with a standard
    deviation of 2.

    :param axes: Axis in which to plot the contour lines.
    :param data: Data for which to plot the heat map. The x values are
        assumed to be in the first column, the y values in the second
        column and the z values in the third column.
    :param lvl: Levels of the added contour lines and labels.
    :param kwargs: Keyword arguments to use for the contour plot.
    :param label_loc: Manually pick locations of the contour labels.
    '''
    if kwargs is None:
        kwargs = {}
    contour = plot_contour_lines(axes, data, smooth_sigma=2, levels=levels,
                                 **kwargs)
    axes.clabel(contour, inline=True, levels=levels, manual=label_loc)
