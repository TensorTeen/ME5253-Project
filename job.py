import sys
import json
from pathlib import Path
from datetime import datetime
import multiprocessing
from multiprocessing.pool import Pool
import logging
import logging.config
from yaml import safe_load
from pathlib import Path
from argparse import ArgumentParser


import numpy as np
import torch

from networked_agents_2.train import train
from networked_agents_2.plots import (
    globally_averaged_plot,
    q_values_plot,
    advantages_plot,
)
from networked_agents_2.plots import delta_plot, mu_plot, pi_plot, log_plot
from networked_agents_2.stats import rel_entropy, ks_test

torch.manual_seed(0)

Path(".logs").mkdir(exist_ok=True)
logging.config.dictConfig(config=safe_load(Path("./logging.yaml").open()))
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def fn(args):
    return train(*args)


# helps transform a list of dictionaries into a pair of lists
def gn(adict, pos):
    return (adict["distributed"][pos], adict["distributed"][pos])


# TODO: change back to centralized (the 1st one)


def unwrap(alist, pos):
    return zip(*[gn(elem, pos) for elem in alist])


def main(n_runs, n_processors, n_steps, n_episodes):
    results_path = Path("data/results")
    results_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H_%M_%S.%f")
    print(f"Experiment timestamp: {timestamp}\n")

    base_args = (n_steps, n_episodes)
    train_args = [base_args + (n_run * 10,) for n_run in range(n_runs)]

    if n_processors > 1:
        pool = Pool(n_processors)
        results = pool.map(fn, train_args)
        pool.close()
        pool.join()
    else:
        results = []
        for args in train_args:
            results.append(train(*args))

    results_path = results_path / timestamp
    results_path.mkdir(exist_ok=True)
    sys.stdout.write(str(results_path))

    # get globally averaged return
    with (results_path / "results.json").open("w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    centralized_J, decentralized_J = unwrap(results, "J")
    globally_averaged_plot(centralized_J, decentralized_J, results_path)

    centralized_Q, decentralized_Q = unwrap(results, "Q")
    q_values_plot(centralized_Q, decentralized_Q, results_path)

    _, decentralized_A = unwrap(results, "A")
    advantages_plot(decentralized_A, results_path)

    centralized_delta, decentralized_delta = unwrap(results, "delta")
    delta_plot(centralized_delta, decentralized_delta, results_path)

    centralized_mu, decentralized_mu = unwrap(results, "mu")
    mu_plot(centralized_mu, decentralized_mu, results_path)

    # TODO: get for non-linear
    # centralized_pi, decentralized_pi = unwrap(results, "pi")
    # pi_plot(centralized_pi, decentralized_pi, results_path)

    centralized_log, decentralized_log = unwrap(results, "data")
    log_plot(centralized_log[0], decentralized_log[0], results_path)

    # centralized_jp, decentralized_jp = unwrap(results, "joint_policy")
    # rel_entropy(centralized_jp[0], decentralized_jp[0])
    # ks_test(centralized_jp[0], decentralized_jp[0])

    return results, str(results_path)

def cli():
    parser = ArgumentParser()
    parser.add_argument("-r", "--runs", type=int, default=1)
    parser.add_argument("-p", "--processors", type=int, default=1)
    parser.add_argument("-s", "--steps", type=int, default=120)
    parser.add_argument("-e", "--episodes", type=int, default=10)
    args = parser.parse_args()

    Path("./data/results").mkdir(exist_ok=True, parents=True)
    results, results_path = main(
        n_runs=args.runs,
        n_processors=args.processors,
        n_steps=args.steps,
        n_episodes=args.episodes,
    )


if __name__ == "__main__":
    cli()
