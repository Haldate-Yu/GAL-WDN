import pandas as pd
import argparse
import warnings
from utils.save_results import processed_result

warnings.filterwarnings("ignore")


def args_init():
    # ----- ----- ----- ----- ----- -----
    # Command line arguments
    # ----- ----- ----- ----- ----- -----
    parser = argparse.ArgumentParser()
    # Running Settings
    parser.add_argument('--run_id',
                        default=1,
                        type=int,
                        help="Run ID.")

    # WDS Loading Settings
    parser.add_argument('--wds',
                        default='anytown',
                        type=str,
                        help="Water distribution system.")
    parser.add_argument('--db',
                        default='doe_pumpfed_1',
                        type=str,
                        help="DB.")
    parser.add_argument('--budget',
                        default=1,
                        type=int,
                        help="Sensor budget.")
    parser.add_argument('--obsrat',
                        default=.05,
                        type=float,
                        help="Observation ratio."
                        )
    parser.add_argument('--adj',
                        default='binary',
                        choices=['binary', 'weighted', 'logarithmic', 'pruned', 'm-GCN'],
                        type=str,
                        help="Type of adjacency matrix.")
    parser.add_argument('--deploy',
                        default='random',
                        choices=['master', 'dist', 'hydrodist', 'hds', 'hdvar', 'random', 'xrandom'],
                        type=str,
                        help="Method of sensor deployment.")

    parser.add_argument('--seed',
                        default=1,
                        type=int,
                        help="seed settings")
    parser.add_argument('--epoch',
                        default=1,
                        type=int,
                        help="Number of epochs.")
    parser.add_argument('--idx',
                        default=None,
                        type=int,
                        help="Dev function.")

    # Model Training Settings
    parser.add_argument('--model',
                        default='ori',
                        type=str,
                        help="Select model.")
    parser.add_argument('--n_layers',
                        default='2',
                        type=int,
                        help="Num of model layers.")
    parser.add_argument('--hidden_dim',
                        default='64',
                        type=int,
                        help="Num of hidden dims.")
    parser.add_argument('--dropout',
                        default=0.1,
                        type=float,
                        help="Dropout rate.")
    parser.add_argument('--use_weight',
                        default=False,
                        type=bool,
                        help="Use Dataset Edge Weight")
    parser.add_argument('--batch',
                        default='40',
                        type=int,
                        help="Batch size.")
    parser.add_argument('--lr',
                        default=0.0003,
                        type=float,
                        help="Learning rate.")
    parser.add_argument('--decay',
                        default=0.000006,
                        type=float,
                        help="Weight decay.")
    parser.add_argument('--tag',
                        default='basic',
                        type=str,
                        help="Custom tag.")

    # M-gcn Settings
    parser.add_argument('--m_gcn_n_hops',
                        default='1',
                        type=int,
                        help="Num of hops in m-gcn.")
    parser.add_argument('--m_gcn_n_layers',
                        default='1',
                        type=int,
                        help="Num of layers in GENConvolution.")

    # SSGC Settings
    parser.add_argument('--alpha',
                        default=0.05,
                        type=float,
                        help='SSGC alpha number')
    parser.add_argument('--aggr_type',
                        default='ssgc',
                        type=str,
                        choices=['sgc', 'ssgc', 'ssgc_no_avg'],
                        help="convolution type.")
    parser.add_argument('--norm_type',
                        default=False,
                        type=bool,
                        help="Use Norm @ output")

    # Sensor Placement Settings
    parser.add_argument('--deterministic',
                        action="store_true",
                        help="Setting random seed for sensor placement.")

    args = parser.parse_args()
    return args


def read_result(result_file):
    data = pd.read_csv(result_file)
    error_all, error_obs, error_hid = data.loc[:, 'test_relative_error_all'], \
        data.loc[:, 'test_relative_error_obs'], \
        data.loc[:, 'test_relative_error_hid']
    return error_all.mean(), error_obs.mean(), error_hid.mean()


if __name__ == '__main__':
    args = args_init()
    deploy = ['master', 'dist', 'hydrodist', 'hds', 'hdvar']

    error_all_list, error_obs_list, error_hid_list = [], [], []
    result_dict = {}
    for this_deploy in deploy:
        ori_run_stamp = args.wds + '-' + this_deploy + '-' + args.model + '-ObsRate' + str(
            args.obsrat) + '-' + args.adj
        save_file_name = ori_run_stamp \
                         + '-LR' + str(args.lr) \
                         + '-WD' + str(args.decay) \
                         + '-Layers' + str(args.n_layers) \
                         + '-Hidden' + str(args.hidden_dim) \
                         + '.csv'

        full_result_file = "./results/{}/{}/{}".format(
            args.wds, args.model, save_file_name)
        # print(f'save result file: {full_result_file}')
        avg_error_all, avg_error_obs, avg_error_hid = read_result(full_result_file)
        error_all_list.append(avg_error_all)
        error_obs_list.append(avg_error_obs)
        error_hid_list.append(avg_error_hid)

        result_dict[this_deploy] = [avg_error_all, avg_error_obs, avg_error_hid]
    result_dict["final"] = [sum(error_all_list) / len(error_all_list),
                            sum(error_obs_list) / len(error_obs_list),
                            sum(error_hid_list) / len(error_hid_list)]
    processed_file_name = args.wds + '-' + args.model + '-ObsRate' + str(
        args.obsrat) + '-' + args.adj + '-LR' + str(args.lr) \
                          + '-WD' + str(args.decay) \
                          + '-Layers' + str(args.n_layers) \
                          + '-Hidden' + str(args.hidden_dim) \
                          + '.csv'
    processed_result(args, processed_file_name, result_dict)
