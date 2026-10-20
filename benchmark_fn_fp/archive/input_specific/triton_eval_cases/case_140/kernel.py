import numpy as np
import torch
import triton
import triton.language as tl

SEED = 734201
N, STEPS = 128, 6


@triton.jit
def _momentum_step(Weights, Velocity, Gradient, N: tl.constexpr):
    index = tl.arange(0, N)
    old_weights = tl.load(Weights + index)
    old_velocity = tl.load(Velocity + index)
    gradient = tl.load(Gradient + index)
    next_velocity = 0.75 * old_velocity + 0.25 * gradient
    next_weights = old_weights - 0.03125 * next_velocity
    tl.store(Velocity + index, next_velocity)
    tl.store(Weights + index, next_weights)


def make_inputs_numpy():
    rng = np.random.Generator(np.random.PCG64(SEED))
    weights = rng.uniform(0.5, 1.5, N).astype(np.float32)
    velocity = rng.normal(0.0, 0.001, N).astype(np.float32)
    gradient_scale = 10.0 ** rng.uniform(-4.0, 2.0)
    persistent = rng.normal(0.0, 1.0, N)
    gradients = (gradient_scale * (
        persistent[None, :] + rng.normal(0.0, 0.25, (STEPS, N)))).astype(np.float32)
    return weights, velocity, gradients


def make_inputs(device="cuda"):
    return tuple(torch.from_numpy(x).to(device) for x in make_inputs_numpy())


def step(weights, velocity, gradient):
    _momentum_step[(1,)](weights, velocity, gradient, N=N,
                        num_warps=1, enable_fp_fusion=False)
    return weights.view(N)


def run_sequence(weights, velocity, gradients):
    history = []
    for gradient in gradients:
        history.append(step(weights, velocity, gradient).detach())
    return history


def run(weights, velocity, gradients):
    return torch.stack(run_sequence(weights, velocity, gradients))


def initial_probe(device="cuda"):
    weights, velocity, gradients = make_inputs(device)
    reference_weights = weights.clone()
    reference_velocity = velocity.clone()
    actual_history, reference_history = [], []
    for gradient in gradients:
        actual_history.append(step(weights, velocity, gradient).detach())
        reference_velocity.copy_(0.75 * reference_velocity + 0.25 * gradient)
        reference_weights.copy_(reference_weights - 0.03125 * reference_velocity)
        reference_history.append(reference_weights.view(N).detach())
    actual, expected = torch.stack(actual_history), torch.stack(reference_history)
    def error(a, b):
        return float((torch.linalg.vector_norm(a.double() - b.double()) /
            torch.clamp(torch.linalg.vector_norm(b.double()), min=0.1)).item())
    history_error = error(actual, expected)
    weights_error = error(weights, reference_weights)
    velocity_error = error(velocity, reference_velocity)
    return {"history_relative_error": history_error,
            "final_weights_relative_error": weights_error,
            "final_velocity_relative_error": velocity_error,
            "tolerance": 0.025, "state_tolerance": 0.00001,
            "passed": bool(history_error <= 0.025 and weights_error <= 0.00001
                           and velocity_error <= 0.00001),
            "observation": "both retained histories stacked after all six updates"}
