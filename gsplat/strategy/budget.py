from dataclasses import dataclass
from typing import Union, Dict, Any
import torch
from .default import DefaultStrategy, reset_opa


@dataclass
class BudgetStrategy(DefaultStrategy):
    """A densification strategy that allows to set a soft target interval to the number of gaussians."""

    min_target: int = 0
    """
    Skips pruning if number of gaussian is under this value. set to less or equal to 0 to always check for pruning.
    If greater than 0, must be less than or equal to max_target.
    """
    max_target: int = 0
    """
    Skips growing (aka clone & split) if number of gaussian is above this value. set to less or equal to 0 to always check for growing.
    If greater than 0, must be greater than or equal to min_target.
    """

    def __post_init__(self):
        # validate min_target and max_target preconditions
        if not (self.min_target <= 0 or self.max_target <= 0) and (self.min_target > self.max_target):
            raise ValueError(
                f"Invalid interval: 0 <= min_target({self.min_target}) <= max_target({self.max_target})"
            )

    def _should_grow(self, n_gaussians: int) -> bool:
        if self.max_target <= 0:
            return True
        return n_gaussians < self.max_target
    
    def _should_prune(self, n_gaussians: int) -> bool:
        if self.min_target <= 0:
            return True
        return n_gaussians > self.min_target

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            if self._should_grow(params["means"].shape[0]):
                # grow GSs
                n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. Now having {len(params['means'])} GSs."
                    )
            elif self.verbose:
                print(
                    f"Step {step}: Too many GSs for duplication and splitting. Now having {params['means'].shape[0]} GSs."
                )

            if self._should_prune(params["means"].shape[0]):
                n_prune = self._prune_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_prune} GSs pruned. Now having {params['means'].shape[0]} GSs."
                    )
            elif self.verbose:
                print(
                    f"Step {step}: Not enough GSs for pruning. Now having {params['means'].shape[0]} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 & step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )


if __name__ == '__main__':
    print("Init Tests...")
    try:
        BudgetStrategy(min_target=10, max_target=100)
        BudgetStrategy(min_target=10, max_target=10)
        BudgetStrategy(min_target=100)
        BudgetStrategy(max_target=100)
    except Exception as e:
        print(f"Test failed: {e}")
        exit(1)

    try:
        BudgetStrategy(min_target=100, max_target=10)
        print("Test failed: Expected ValueError")
        exit(1)
    except ValueError as e:
        pass

    print("_should_(grow|prune) Tests...")
    def run(s, n_gaussians, expected):
        r = s._should_grow(n_gaussians), s._should_prune(n_gaussians)
        if r != expected:
            raise AssertionError(f"{r} != {expected}")
    s = BudgetStrategy(min_target=10)
    run(s, 0, (True, False))
    run(s, 10, (True, False))
    run(s, 100, (True, True))

    print("All tests successful")