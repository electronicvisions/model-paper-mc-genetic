#!/usr/bin/env python3
"""
Plotting functions generating all plots of the paper.

The data needed for this can be generated using the script
:py:mod:`record_experiment`.
"""
from functools import partial
import quantities as pq

from model_paper_mc_genetic.helpers import name_wrapper

from model_paper_mc_genetic.plotting import PaperPlots, set_latex_style


def main(text_width: float,
         data_path: str,
         save_path: str,
         file_extension: str,
         latex: bool) -> None:
    '''
    Plot experiments of paper using the provided data.

    :param text_width: Width of the generated plots in mm.
    :param data_path: Path containing the data needed for the plotting.
    :param save_path: Path to which the plots will be saved to.
    :param file_extension: File extension for the generated plots.
    :param latex: If true the plots will be rendered using the latex style
        provided by the set_latex_style method.
    '''
    data_wrapper = partial(name_wrapper, savepath=data_path, fileextension='')
    save_wrapper = partial(name_wrapper, savepath=save_path,
                           fileextension=file_extension)

    if latex:
        set_latex_style()

    figure_width_inch = (text_width * pq.mm).rescale(pq.inch).magnitude
    figure_height_inch = figure_width_inch / 2

    paper_plots = PaperPlots(figure_width_inch, figure_height_inch,
                             data_wrapper, save_wrapper)
    # Plotting functions are ordered as the plots appear in the paper
    paper_plots.plot_single_trace_attenuation()
    paper_plots.plot_single_trace_exp_decay()
    paper_plots.plot_histograms()
    paper_plots.plot_sta()
    paper_plots.plot_grid_search()

    # Genetic algorithm experiments
    paper_plots.plot_genetic_algorithm_performance()
    paper_plots.plot_genetic_algorithm_results()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Plot the data from 'record_experiment.py', which results "
                    "in the paper plots.")
    parser.add_argument("-text_width",
                        help="Column width of the text in mm used in the "
                             "paper. The default is the text column width in "
                             "Open Research Europe Articles.",
                        type=float,
                        default=83.82397)
    parser.add_argument("-file_extension",
                        help="Specify the file format which the plots are "
                             "saved in. Choose between `.png` and `.pgf`.",
                        type=str,
                        choices=['.png', '.pgf'],
                        default='.png')
    parser.add_argument("-data_path",
                        help="Path to the directory where the data is stored.",
                        type=str,
                        default='')
    parser.add_argument("-save_path",
                        help="Path to the directory where the plots are "
                             "saved.",
                        type=str,
                        default='')
    parser.add_argument("--latex",
                        help="The plots will be rendered using the latex style"
                             " provided by set_latex_style.",
                        action="store_true")

    args = parser.parse_args()

    main(text_width=args.text_width,
         data_path=args.data_path,
         save_path=args.save_path,
         file_extension=args.file_extension,
         latex=args.latex)
