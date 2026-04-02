"""
Exercise 2: Implement and Verify the STE — SOLUTION
====================================================
Course: Self-Compressing Neural Networks
Module: 1 — The Straight-Through Estimator
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Implementation 1: torch.autograd.Function
# ---------------------------------------------------------------------------

class StraightThroughRound(torch.autograd.Function):
    """Custom autograd Function implementing the Straight-Through Estimator.

    Forward:  returns x.round()   (exact rounding)
    Backward: returns grad_output  (identity — gradient passes through unchanged)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Round x to nearest integer."""
        # Optionally save for inspection (not needed for STE backward)
        ctx.save_for_backward(x)
        return x.round()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Straight-Through: pass gradient through unchanged."""
        return grad_output


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Functional wrapper for StraightThroughRound."""
    return StraightThroughRound.apply(x)


# ---------------------------------------------------------------------------
# Implementation 2: inline detach trick
# ---------------------------------------------------------------------------

def ste_round_inline(x: torch.Tensor) -> torch.Tensor:
    """STE via the inline detach trick.

    Forward:  x.round()  (true rounding)
    Backward: identity   (STE)

    Derivation:
      f(x) = (x.round() - x).detach() + x

      Forward: (r - x).detach() + x = (r - x) + x = r  where r = x.round()
      Backward: df/dx = d/dx[(const) + x] = 1
                (detach makes (x.round() - x) a constant in backward)
    """
    return (x.round() - x).detach() + x


# ---------------------------------------------------------------------------
# Gradient verification
# ---------------------------------------------------------------------------

def verify_gradient_flow(x: torch.Tensor) -> dict:
    """Verify gradient flow through ste_round.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (values, no grad needed — cloned internally).

    Returns
    -------
    dict with forward_correct, grad_is_nonzero, grad_is_identity,
         forward_values, grad_values.
    """
    # Fresh leaf tensor with gradient tracking
    x_leaf = x.clone().detach().requires_grad_(True)

    # Forward pass
    y = ste_round(x_leaf)

    # Loss: sum of all elements (gradient should be all ones)
    loss = y.sum()

    # Backward pass
    loss.backward()

    forward_values = y.detach()
    grad_values = x_leaf.grad.clone()

    return {
        "forward_correct":  torch.allclose(forward_values, x.round()),
        "grad_is_nonzero":  grad_values.abs().sum() > 0,
        "grad_is_identity": torch.allclose(grad_values, torch.ones_like(x_leaf)),
        "forward_values":   forward_values,
        "grad_values":      grad_values,
    }


# ---------------------------------------------------------------------------
# Main (DO NOT MODIFY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("Exercise 2: Implement and Verify the STE")
    print("=" * 60)
    print()

    test_inputs = [
        torch.tensor([0.3, 0.7, -0.3, -0.7, 1.4, -1.4, 2.6, -2.6]),
        torch.tensor([0.49, 0.51, -0.49, -0.51]),
        torch.randn(16),
    ]

    print("TEST 1: Forward correctness")
    print("-" * 40)
    all_forward_pass = True
    for i, x in enumerate(test_inputs):
        y_ste     = ste_round(x)
        y_inline  = ste_round_inline(x)
        y_true    = x.round()

        ste_match    = torch.allclose(y_ste, y_true)
        inline_match = torch.allclose(y_inline, y_true)

        if not (ste_match and inline_match):
            all_forward_pass = False

        print(f"  Input {i+1}:  autograd match={ste_match}  inline match={inline_match}")

    assert all_forward_pass, "Forward values don't match torch.round()!"
    print("  ✓ Both implementations match torch.round() exactly")
    print()

    print("TEST 2: Gradient flow through STE")
    print("-" * 40)

    x_test = torch.tensor([0.3, 0.7, -0.4, 1.6])
    result = verify_gradient_flow(x_test)

    print(f"  Input:            {x_test.tolist()}")
    print(f"  Forward output:   {result['forward_values'].tolist()}")
    print(f"  Expected:         {x_test.round().tolist()}")
    print(f"  Gradient values:  {result['grad_values'].tolist()}")
    print(f"  Expected grad:    [1.0, 1.0, 1.0, 1.0]")
    print()
    print(f"  forward_correct:  {result['forward_correct']}")
    print(f"  grad_is_nonzero:  {result['grad_is_nonzero']}")
    print(f"  grad_is_identity: {result['grad_is_identity']}")

    assert result["forward_correct"]
    assert result["grad_is_nonzero"]
    assert result["grad_is_identity"]
    print("  ✓ Gradient flows correctly as identity (STE working)")
    print()

    print("TEST 3: autograd.Function vs inline trick — must be identical")
    print("-" * 40)

    for i, x_base in enumerate(test_inputs):
        x1 = x_base.clone().detach().requires_grad_(True)
        x2 = x_base.clone().detach().requires_grad_(True)

        y1 = ste_round(x1)
        y2 = ste_round_inline(x2)

        (y1 * torch.arange(1, len(x_base) + 1, dtype=torch.float)).sum().backward()
        (y2 * torch.arange(1, len(x_base) + 1, dtype=torch.float)).sum().backward()

        values_match = torch.allclose(y1, y2)
        grads_match  = torch.allclose(x1.grad, x2.grad)

        print(f"  Input {i+1}:  values_match={values_match}  grads_match={grads_match}")
        assert values_match
        assert grads_match

    print("  ✓ Both implementations produce identical forward values AND gradients")
    print()

    print("TEST 4: Numerical gradient verification")
    print("-" * 40)

    x_num = torch.tensor([0.3, 0.8, 1.6, -0.4], dtype=torch.float64)
    x_num.requires_grad_(True)

    y = ste_round(x_num)
    loss = (y * torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)).sum()
    loss.backward()
    analytical_grad = x_num.grad.clone()

    expected_grad = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    numerical_match = torch.allclose(analytical_grad, expected_grad)
    print(f"  Analytical STE grad: {analytical_grad.tolist()}")
    print(f"  Expected (weights):  {expected_grad.tolist()}")
    print(f"  Match: {numerical_match}")
    assert numerical_match
    print("  ✓ STE gradient equals incoming gradient (identity pass confirmed)")
    print()

    print("=" * 60)
    print("ALL STE TESTS PASSED")
    print("=" * 60)
    print()
    print("Both STE implementations are correct and identical.")
    print("autograd.Function: explicit forward/backward, pedagogically clear")
    print("inline trick:      (x.round()-x).detach()+x — used in the paper")
