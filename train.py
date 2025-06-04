import numpy as np
import time
import hydra
from omegaconf import DictConfig
from common import get_args, experiment_setup
# from scripts.reactive_tamp import REACTIVE_TAMP
# from scripts.sim import run_sim
from m3p2i_aip.config.config_store import ExampleConfig

import json
import logging
import numpy as np

import torch, hydra, zerorpc

def convert_ndarray(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_ndarray(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: convert_ndarray(v) for k, v in obj.items()}
    elif isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, list) and all(isinstance(i, torch.Tensor) for i in x):
        return [i.tolist() for i in obj]

    return obj

def main():
    args = get_args()
    env, env_test, agent, buffer, learner, tester = experiment_setup(args)

    planner = zerorpc.Client()
    planner.connect("tcp://127.0.0.1:4242")
    print("Server found and wait for the viewer")

    args.epochs = args.epochs + 10
    training_state = {
        "total_epochs": args.epochs,
    }

    for epoch in range(args.epochs):
        print("*************************epoch***********************", epoch, args.epochs)
        training_state["current_epoch"] = epoch
        for cycle in range(args.cycles+5):
            print("*********************************cycle*******************************", cycle, args.cycles)
            args.logger.tabular_clear()
            start_time = time.time()

            learner.learn(args, env, env_test, agent, buffer, planner, training_state)

            log_entry = {
                "epoch": epoch,
                "cycle": cycle,
                "initial_goals": convert_ndarray(learner.initial_goals),
                "desired_goals": convert_ndarray(learner.desired_goals),
                "explore_goals": convert_ndarray(learner.explore_goals),
                "trajectories": convert_ndarray(learner.achieved_trajectories),
                "episode_return": convert_ndarray(learner.episode_return),
            }
            with open("explore_goals4.json", "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        tester.epoch_summary()

    tester.final_summary()

if __name__ == "__main__":
    main()

