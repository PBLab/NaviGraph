from omegaconf import OmegaConf
from pandas import read_pickle
from pandas.testing import assert_frame_equal

from navigraph.session_manager import Manager


class TestAnalyzer:
    """
        Test Analyzer class
    """
    _TEST_CONFIG_PATH = './configs/test_config.yaml'
    _TEST_DF_PKL_PATH = './data/test_df.pkl'

    def test_compare_analysis_results(self):
        cfg = OmegaConf.load(TestAnalyzer._TEST_CONFIG_PATH)
        test_df = read_pickle(TestAnalyzer._TEST_DF_PKL_PATH)

        manager = Manager(cfg=cfg)
        manager.run()
        df = manager.analysis_df

        assert_frame_equal(test_df, df)