import unittest

from model_paper_mc_genetic.scripts.record_experiment import main
# from model_paper_mc_genetic.scripts.plot_experiment import main as plot


class TestExperiment(unittest.TestCase):
    def test_00_minimal_experiment(self):
        '''
        Test minimal experiment without grid search and genetic algorithm.
        '''
        self.assertIsNone(main(save_path=''))

    def test_01_experiment(self):
        '''
        Test experiment with grid search and genetic algorithm.
        '''
        self.assertIsNone(main(save_path='test_results',
                               grid_search=True,
                               conductance=[10, 1020, 5],
                               genetic_algorithm=True,
                               repetitions=1))

    # def test_02_visualization(self):
    #     '''
    #     Test visualization functions.
    #     '''
    #     # TODO: can not work currently since we expect 10 repetitions for the
    #     # genetic algorithm...
    #     self.assertIsNone(plot(text_width=84,
    #                            data_path='test_results',
    #                            save_path='test_results',
    #                            file_extension='.png',
    #                            latex=False))


if __name__ == "__main__":
    unittest.main()
