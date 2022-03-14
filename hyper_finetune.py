# Implementation of hyper-finetune algorithm
import argparse
import numpy as np
from argparse import Namespace
import threading
from params import N_EXPERIMENTS
from main import main
import torch.multiprocessing as mp
from queue import Queue


def test_all(args, search_keys, search_vals_list, training, num_workers):
    arg_dict = vars(args)
    ctx = mp.get_context()
    args_new = ctx.Queue()
    q_lock = ctx.Lock()
    for idx, sub_args in enumerate(search_vals_list):
        # substitute args
        for i, key in enumerate(search_keys):
            arg_dict[key] = sub_args[i]
        args_new.put((idx, Namespace(**arg_dict)))
    threads = []
    metrics = ctx.Array("f", len(search_vals_list))
    def training_process(args_new, q_lock, metrics):
        while not args_new.empty():
            with q_lock:
                idx, args = args_new.get()
            result = training(args)[-1]
            with q_lock:
                metrics[idx] = result
    for thread_id in range(num_workers):
        threads.append(ctx.Process(target=training_process, args=(args_new, q_lock, metrics)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Return best arguments and metric
    metrics = np.array(metrics)
    max_i = np.argmax(metrics)
    best_args = dict(zip(search_keys, search_vals_list[max_i]))
    print("All result", list(zip(search_vals_list, metrics)))
    print("Result:", best_args, metrics[max_i])
    return best_args, metrics[max_i]


def grid_search(args, search_space, training):
    """
    args: arguments need for training
    search_space: dict: key is argument-name and value is list of this argument
    training: training procedure receive args and return a metric
    """
    # Generate all argument combinations
    def arg_combine(vlista, valueb):
        return list(tuple(x) + (y, ) for x in vlista for y in valueb)
    all_args = [()]
    for vlist in search_space.values():
        all_args = arg_combine(all_args, vlist)
    return test_all(args, search_space.keys(), all_args, training, args.n_workers)


def random_search(args, cat_args, cat_list, cont_args, cont_bounds, training, n_tests, sample_method=None):
    # Perform random_search for given times
    # Generate arguments
    all_args = []
    if sample_method == None:
        sample_method = [None] * len(cont_args)
    for i in range(n_tests):
        # generate categorical args
        sub_args = []
        for arglist in cat_list:
            sub_args.append(arglist[np.random.choice(len(arglist))])
        for (lower, upper), sampler in zip(cont_bounds, sample_method):
            if not sampler:
                # Use a uniform sampler
                sub_args.append(float(np.random.rand(1)) * (upper-lower) + lower)
            elif sampler == "loguni":
                value = float(np.random.rand(1)) * (np.log(upper)-np.log(lower)) + np.log(lower)
                sub_args.append(np.exp(value))
            else:
                raise ValueError()
        all_args.append(sub_args)
    return test_all(args, cat_args+cont_args, all_args, training, args.n_workers)

         

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=str, default="result.npy")
    parser.add_argument("--n-exp", type=int, default=N_EXPERIMENTS)
    parser.add_argument("--verbose", "-v", action='store_true')
    parser.add_argument("--use-fc", action="store_true")
    parser.add_argument("--n-cls-start", type=int, default=5)
    parser.add_argument("--n-cls-end", type=int, default=50)
    parser.add_argument("--finetune-backbone", action="store_true")
    parser.add_argument("--use-adv-baseline", action="store_true")
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args()
    search_space = {"lr": [1e-3, 1e-4],
                    "weight_decay": [0, 1e-4], 
                    "batch_size":[1, 5], 
                    "use_fc": [True, False]}
    grid_search(args, search_space, main)