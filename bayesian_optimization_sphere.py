import os

import matplotlib.pyplot as plt
from ioh import get_problem, ProblemClass
from ioh.iohcpp.logger.trigger import Each, ON_IMPROVEMENT

from Algorithms import Vanilla_BO, BOTorchImplementation
import multiprocessing as mp
from itertools import product
import torch
from ioh import logger  # Add this import
import ioh.iohcpp.logger as logger_lib


def run_optimization_on_instance(args, logger):
    """
    Runs Bayesian Optimization on the Sphere function instance.
    Modified to accept a single tuple of arguments for multiprocessing.

    Args:
        args: Tuple of (instance_id, seed, budget, n_DoE)
    """
    instance_id, seed, budget, n_DoE = args

    # Retrieve the Sphere problem (5-dimensional)
    problem = get_problem("Sphere", instance=instance_id, dimension=5, problem_class=ProblemClass.BBOB)

    problem.attach_logger(logger)

    # Instantiate the Bayesian optimizer
    optimizer = BOTorchImplementation(random_seed=seed, budget=budget, n_DoE=n_DoE)

    # Reset the optimizer
    #optimizer.reset()

    logger.watch(optimizer,"acquistion_function_name")

    # Run the optimizer
    optimizer(problem=problem)

    return (instance_id, seed, optimizer.f_evals)


def main():
    # Configuration
    num_instances = 2
    num_repetitions = 2
    budget = 30
    n_DoE = 10

    triggers = [
        Each(10),  # Log after (10) evaluations
        ON_IMPROVEMENT  # Log when there's an improvement
    ]

    # Set up IOH logger
    l = logger.Analyzer(
        triggers=triggers,
        root=os.getcwd(),
        folder_name="bayesian_opt_results",  # Name for the experiment folder
        algorithm_name="Bayesian-Optimization",  # Name of your algorithm
        algorithm_info="BO with botorch",  # Additional information about the algorithm
        additional_properties= [logger_lib.property.RAWYBEST
                                ], # Use this to log the best-so-far
        store_positions=True               # store x-variables in the logged files
    )

    # Create all combinations of instances and repetitions
    tasks = [
        (inst, 1000 * inst + rep, budget, n_DoE)
        for inst, rep in product(
            range(1, num_instances + 1),
            range(1, num_repetitions + 1)
        )
    ]

    # Attach logger to the problem
    #problem = get_problem("Sphere", instance=1, dimension=5, problem_class=ProblemClass.BBOB)

    # problem = get_problem(22,  # An integer denoting one of the 24 BBOB problem
    #                       instance=1,
    #                       # An instance, meaning the optimum of the problem is changed via some transformations
    #                       dimension=5,  # The problem's dimension
    #                       )
    # problem.attach_logger(l)

    # Run optimizations sequentially
    results = [run_optimization_on_instance(task, l) for task in tasks]

    l.close()

    # Create figure for plotting
    fig, axes = plt.subplots(nrows=num_instances, ncols=num_repetitions,
                             figsize=(20, 30), sharex=True, sharey=True)

    # Process results and create plots
    for instance_id, seed, performance in results:
        # Calculate row and column indices
        rep = (seed % 1000)  # Extract repetition number from seed
        row = instance_id - 1
        col = rep - 1

        # Generate x-axis values
        iterations = range(1, len(performance) + 1)

        # Plot performance
        ax = axes[row][col]
        ax.plot(iterations, performance, marker='o', markersize=3, linestyle='-')
        ax.set_yscale('log')
        ax.set_title(f"Inst {instance_id}, Rep {rep}", fontsize=10)
        ax.set_xlabel("Iteration", fontsize=8)
        ax.set_ylabel("f* - f", fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()