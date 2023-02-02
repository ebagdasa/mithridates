import numpy as np

from ray import tune

from mnist_example import run_training


search_space = {
    "poison_ratio": tune.loguniform(0.00001, 0.1),
}


def tune_run(params):
    fixed_params = {
        "lr": 1.0,  # tune.sample_from(lambda spec: 10 ** (-10 * np.random.rand())),
        "batch_size": 64,
        "test_batch_size": 1000,
        "epochs": 14,
        "gamma": 0.7,
        "no_cuda": False,
        "no_mps": False,
        "seed": 1,
        "print": False
    }
    params.update(fixed_params)
    accuracy, backdoor_accuracy = run_training(params)
    tune.report(accuracy=accuracy, backdoor_accuracy=backdoor_accuracy)


tuner = tune.Tuner(
    tune_run,
    param_space=search_space,
    tune_config=tune.TuneConfig(num_samples=10)
)
results = tuner.fit()