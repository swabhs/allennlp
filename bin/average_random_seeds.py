
import glob
import os
import argparse
import json
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_root', help='Description of language model', default=''
    )

    args = parser.parse_args()
    save_root = args.save_root

    seed_dirs = glob.glob(save_root + '_SEED_*')
    val = []
    test = []
    for seed_dir in seed_dirs:
        with open(os.path.join(seed_dir, 'config.json'), 'r') as fin:
            config = json.load(fin)
        metric = config['trainer']['validation_metric'][1:]

        with open(os.path.join(seed_dir, 'metrics.json'), 'r') as fin:
            results = json.load(fin)

        val_metric = results['best_validation_' + metric]
        val.append(val_metric)
        # try:
        #     test_metric = results['test_' + metric]
        #     test.append(test_metric)
        # except KeyError:
        #     pass

        try:
            with open(os.path.join(seed_dir, 'evaluation_metrics.json'), 'r') as fin:
                eval_results = json.load(fin)

            test_metric = eval_results[metric]
            test.append(test_metric)
        except (KeyError, OSError) as e:
            pass


    # now compute mean, std, max, min
    for i, metrics in enumerate([val, test]):
        if i == 0:
            print("VALIDATION")
        elif i==1 and len(metrics) == 0:
            print("NO TEST")
            continue
        else:
            print("TEST")

        mm = np.array(metrics) * 100
        m = np.mean(mm)
        s = np.std(mm)
        mn = np.min(mm)
        mx = np.max(mm)
        fmt = "{:.2f}"
        ss = fmt + ' +/- ' + fmt + ' [' + fmt + ', ' + fmt + ']\n'
        print(ss.format(m, s, mn, mx))

