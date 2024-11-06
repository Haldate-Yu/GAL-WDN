import csv
import os


def save_results(args, file_name, tst_loss, tst_rel_err, tst_rel_err_obs, tst_rel_err_hid):
    if not os.path.exists('./results/{}'.format(args.wds)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./results/{}'.format(args.wds))
    if not os.path.exists('./results/{}/{}'.format(args.wds, args.model)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./results/{}/{}'.format(args.wds, args.model))

    filename = "./results/{}/{}/{}".format(
        args.wds, args.model, file_name)

    if args.model == 'ssgc':
        headerList = ["Method", "Seed", "Learning_Rate",
                      "Weight_Decay", "Observation_Ratio",
                      "Alpha", "Encoder_Layers",
                      "Hidden_Dims",
                      "::::::::",
                      "test_loss", "test_relative_error_all",
                      "test_relative_error_obs", "test_relative_error_hid"]
    elif args.model == 'm_gcn':
        headerList = ["Method", "Seed", "Learning_Rate",
                      "Weight_Decay", "Observation_Ratio",
                      "Encoder_Layers", "N_Hops",
                      "GEN_Layers", "Hidden_Dims",
                      "::::::::",
                      "test_loss", "test_relative_error_all",
                      "test_relative_error_obs", "test_relative_error_hid"]
    else:
        headerList = ["Method", "Seed", "Learning_Rate",
                      "Weight_Decay", "Observation_Ratio",
                      "Encoder_Layers", "Hidden_Dims",
                      "::::::::",
                      "test_loss", "test_relative_error_all",
                      "test_relative_error_obs", "test_relative_error_hid"]

    with open(filename, "a+") as f:
        f.seek(0)
        header = f.read(6)
        if header != "Method":
            dw = csv.DictWriter(f, delimiter=',',
                                fieldnames=headerList)
            dw.writeheader()

        if args.model == 'ssgc':
            line = "{}, {}, {}, {}, {}, {}, {}, {}, :::::::::, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                args.model, args.seed, args.lr, args.decay, args.obsrat,
                args.alpha, args.n_layers, args.hidden_dim,
                tst_loss, tst_rel_err,
                tst_rel_err_obs, tst_rel_err_hid
            )
        elif args.model == 'm_gcn':
            line = "{}, {}, {}, {}, {}, {}, {}, {}, {}, :::::::::, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                args.model, args.seed, args.lr, args.decay, args.obsrat,
                args.n_layers, args.m_gcn_n_hops,
                args.m_gcn_n_layers, args.hidden_dim,
                tst_loss, tst_rel_err,
                tst_rel_err_obs, tst_rel_err_hid
            )
        else:
            line = "{}, {}, {}, {}, {}, {}, {}, :::::::::, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(
                args.model, args.seed, args.lr, args.decay,
                args.obsrat, args.n_layers, args.hidden_dim,
                tst_loss, tst_rel_err,
                tst_rel_err_obs, tst_rel_err_hid
            )

        f.write(line)


def processed_result(args, file_name, result_dict):
    if not os.path.exists('./processed_results/{}'.format(args.wds)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./processed_results/{}'.format(args.wds))
    if not os.path.exists('./processed_results/{}/{}'.format(args.wds, args.model)):
        print("=" * 20)
        print("Creating Results File !!!")

        os.makedirs('./processed_results/{}/{}'.format(args.wds, args.model))

    filename = "./processed_results/{}/{}/{}".format(
        args.wds, args.model, file_name)

    header_list = ["Deploy Style", "Error All", "Error Obs", "Error Hid"]
    record_list = ['master', 'dist', 'hydrodist', 'hds', 'hdvar', 'final']

    for record in record_list:
        with open(filename, "a+") as f:
            f.seek(0)
            header = f.read(6)
            if header != "Deploy":
                dw = csv.DictWriter(f, delimiter=',',
                                    fieldnames=header_list)
                dw.writeheader()

            avg_error_all, avg_error_obs, avg_error_hid = result_dict[record]
            line = "{}, {}, {}, {}\n".format(record, avg_error_all, avg_error_obs, avg_error_hid)
            f.write(line)
