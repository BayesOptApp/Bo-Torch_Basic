from typing import Union, Callable, Optional
from ioh.iohcpp.problem import RealSingleObjective
import numpy as np
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound, \
    AnalyticAcquisitionFunction
from botorch.optim import optimize_acqf
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel
from ..AbstractBayesianOptimizer import AbstractBayesianOptimizer

ALLOWED_ACQUISITION_FUNCTIONS = {
    "EI": ("expected_improvement", ExpectedImprovement),
    "PI": ("probability_of_improvement", ProbabilityOfImprovement),
    "UCB": ("upper_confidence_bound", UpperConfidenceBound)
}


class BOTorchImplementation(AbstractBayesianOptimizer):
    def __init__(self, budget: int, n_DoE: int = 0, acquisition_function: str = "EI",
                 random_seed: int = 43, **kwargs):
        super().__init__(budget, n_DoE, random_seed, **kwargs)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        self.config = {
            "num_restarts": kwargs.get("num_restarts", 10),
            "raw_samples": kwargs.get("raw_samples", 512),
            "batch_size": kwargs.get("batch_size", 1)
        }

        self._model = None
        self._acq_function = None
        self.set_acquisition_function(acquisition_function)

    def assign_new_best(self) -> None:
        """
        Update the current best solution based on all evaluations.
        """
        if self.maximisation:
            self.current_best_index = np.argmax(self.f_evals)
        else:
            self.current_best_index = int(np.argmin(self.f_evals))

        self.current_best = self.f_evals[self.current_best_index]

    def set_acquisition_function(self, acq_func_name: str) -> None:
        """Set the acquisition function based on name."""
        if acq_func_name not in ALLOWED_ACQUISITION_FUNCTIONS:
            raise ValueError(f"Acquisition function must be one of {list(ALLOWED_ACQUISITION_FUNCTIONS.keys())}")

        self.acq_func_name, self.acq_func_class = ALLOWED_ACQUISITION_FUNCTIONS[acq_func_name]

    def _initialize_model(self) -> None:
        """Initialize or update the Gaussian Process model."""
        train_x = torch.from_numpy(np.array(self.x_evals)).to(device=self.device, dtype=self.dtype)
        train_y = torch.from_numpy(np.array(self.f_evals).reshape(-1, 1)).to(device=self.device, dtype=self.dtype)
        bounds = torch.from_numpy(self.bounds.T).to(device=self.device, dtype=self.dtype)

        self._model = SingleTaskGP(
            train_X=train_x,
            train_Y=train_y,
            covar_module=MaternKernel(nu=2.5),
            outcome_transform=Standardize(m=1),
            input_transform=Normalize(
                d=train_x.shape[-1],
                bounds=bounds
            )
        )

    def _optimize_acquisition_function(self) -> torch.Tensor:
        """Optimize the acquisition function to get next sampling point."""
        bounds = torch.from_numpy(self.bounds.T).to(device=self.device, dtype=self.dtype)

        acq_function = self.acq_func_class(
            model=self._model,
            best_f=self.current_best if self.maximisation else -self.current_best,
            maximize=self.maximisation
        )

        candidates, _ = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=self.config["batch_size"],
            num_restarts=self.config["num_restarts"],
            raw_samples=self.config["raw_samples"],
            options={"batch_limit": 5, "maxiter": 200},
        )

        return candidates.detach()

    def __call__(self, problem: Union[RealSingleObjective, Callable],
                 dim: Optional[int] = -1,
                 bounds: Optional[np.ndarray] = None,
                 **kwargs) -> None:

        super().__call__(problem, dim, bounds, **kwargs)
        self._initialize_model()

        for _ in range(self.budget - self.n_DoE):
            # Get next point to evaluate
            new_x = self._optimize_acquisition_function()
            new_x_np = new_x.cpu().numpy().reshape(-1)

            # Evaluate the function
            new_y = problem(new_x_np)

            # Update data
            self.x_evals.append(new_x_np)
            self.f_evals.append(new_y)
            self.number_of_function_evaluations += 1

            # Update best solution
            self.assign_new_best()

            if self.verbose:
                print(f"Current Best: x:{self.x_evals[self.current_best_index]} y:{self.current_best}")

            # Update the model
            self._initialize_model()

        print("Optimization completed!")

    def reset(self) -> None:
        super().reset()
        self._model = None
        self._acq_function = None