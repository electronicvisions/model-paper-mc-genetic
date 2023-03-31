# Description

Experiment and visualization code for the paper: "Parametrizing Analog Multi-Compartment Neurons with Genetic Algorithms" (in review).

The main part of the experiment is to find the appropriate inter-compartment conductance and leak conductance on the neuromorphic computing platform BrainScaleS-2 using genetic algorithms to replicate a desired attenuation behavior of an excitatory post synaptic potential propagating along a linear chain of compartments.

You can access BrainScaleS-2 via [EBRAINS](https://electronicvisions.github.io/documentation-brainscales2/latest/brainscales2-demos/tutorial.html).
There you can also find a simple [tutorial](https://electronicvisions.github.io/documentation-brainscales2/latest/brainscales2-demos/ts_04-mc_genetic_algorithms.html) on how to use genetic algorithms to parameterize BrainScaleS-2.

## Functionality

All experiments of the paper are summerized in one script and can be executed via (assuming you have access to BrainScaleS-2):

```
python3 src/py/model_paper_mc_genetic/scripts/record_experiment.py --grid_search --genetic_algorithm
```

Using the recorded data you can visualize it by executing:

```
python3 src/py/model_paper_mc_genetic/scripts/plot_experiment.py --latex
```

If you don't have access to BrainScaleS-2 you can also download the data of the publication from [heiDATA](https://doi.org/10.11588/data/U2U1IB) and replicate the figures of the paper using the previous command.

## Structure

In `src/py/model_paper_mc_genetic/`

* `scripts/record_experiment.py` executes all experiments of the paper.
* `scripts/plot_experiment.py` visualizes the data recorded in `record_experiment.py`
* `scripts/attenuation_ga.py` provides the genetic algorithm applied to the attenuation experiment.
* `plotting.py` helper functions for plotting the data.
* `helpers.py` further helper functions.
