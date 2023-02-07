import numpy as np
import argparse

from ray import tune
from ray import air

from mnist_example import run_training


def tune_run(params):

    accuracy, backdoor_accuracy = run_training(params)
    tune.report(accuracy=accuracy, backdoor_accuracy=backdoor_accuracy,
                poison_ratio=100*params['poison_ratio'])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ray Tuning')
    parser.add_argument('--run_name', required=True, type=str)

    parser.add_argument('--build_curve', action='store_true')
    parser.add_argument('--run_stage1', default=None, type=str)
    parser.add_argument('--run_stage2', default=None, type=str)
    parser.add_argument('--run_stage3', default=None, type=str)
    parser.add_argument('--run_stage4', default=None, type=str)
    parser.add_argument('--target_ratio', default=None, type=float)
    args = parser.parse_args()

    if args.build_curve:
        print("Building curve with default hyperparameters.")
        params = {
            "poison_ratio": tune.loguniform(0.00001, 0.05),
            "lr": 1.0,  # tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
            "batch_size": 64,
            "test_batch_size": 1000,
            "epochs": 3,
            "gamma": 0.7,
            "no_cuda": False,
            "no_mps": False,
            "seed": 1,
            "print": False
        }
        tuner = tune.Tuner(
            tune.with_resources(tune_run,
                resources={"cpu": 2, "gpu": 0.5}),
            param_space=params,
            tune_config=tune.TuneConfig(num_samples=40),
            run_config=air.RunConfig(name=args.run_name)
        )
        results = tuner.fit()
