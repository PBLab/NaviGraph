import argparse


def get_input_arguments():
    parser = argparse.ArgumentParser(prog='NaviGraph',
                                     description='Complete pipeline for maze experiment analysis in both real world '
                                                 'and frame coordinates')
    parser.add_argument('--config_path',
                        dest='config_path',
                        type=str,
                        help='Path to run time configuration',
                        default="./configs")
    parser.add_argument('--config_name',
                        dest='config_name',
                        type=str,
                        help='Run time configuration file name',
                        default="navigraph_basic")
    args = parser.parse_args()

    return args.config_path, args.config_name


