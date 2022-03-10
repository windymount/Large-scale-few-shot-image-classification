# Implementation of hyper-finetune algorithm
import numpy as np
from argparse import Namespace
import threading
from test import main
from queue import Queue


def test_all(args, search_keys, search_vals_list, training, num_workers):
    arg_dict = vars(args)
    args_new = Queue()
    q_lock = threading.Lock()
    for sub_args in search_vals_list:
        # substitute args
        for i, key in enumerate(search_keys):
            arg_dict[key] = sub_args[i]
        args_new.put(Namespace(**arg_dict))
    threads = []
    output_args, metrics = [], []
    def training_process():
        with q_lock:
            args = args_new.get()
        result = training(args)
        with q_lock:
            output_args.append(args)
            metrics.append(result)
    for thread_id in range(num_workers):
        threads.append(threading.Thread(target=training_process))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    # Return best arguments and metric
    metrics = np.array(metrics)
    max_i = np.argmax(metrics)
    best_args = dict(zip(search_keys, search_vals_list[max_i]))
    return best_args, metrics[max_i]


def gridsearch(args, search_space, training):
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
    return test_all(args, search_space.keys(), all_args, training)


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
    return test_all(args, cat_args+cont_args, all_args, training)

         

if __name__ == "__main__":
    s_space = {"a":[3,5,7], "b":[1, 3], "c":[4, 2]}
    gridsearch(0, s_space, 0)