"""
Exercise 2: Implement and Verify the Straight-Through Estimator
===============================================================
Course: Self-Compressing Neural Networks
Module: 1 — The Straight-Through Estimator

GOAL
----
Implement the STE two ways and verify they are numerically identical:

  1. torch.autograd.Function  — the explicit, textbook approach.
     Forward: x.round(); Backward: return grad_output unchanged.

  2. Inline detach trick       — the one-liner used in the self-compressing
     paper: (x.round() - x).detach() + x

You will also implement verify_gradient_flow() to confirm that gradients
propagate correctly through both implementations.

WHAT TO IMPLEMENT
-----------------
  - StraightThroughRound.forward(ctx, x)   — save and round
  - StraightThroughRound.backward(ctx, grad_output)  — identity pass
  - ste_round_inline(x)   — the one-liner detach trick
  - verify_gradient_flow(x)  — build graph, backprop, check grads

DEPENDENCIES
------------
  pip install torch  # PyTorch >= 2.0
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Implementation 1: torch.autograd.Function
# ---------------------------------------------------------------------------

class StraightThroughRound(torch.autograd.Function):
    """Custom autograd Function implementing the Straight-Through Estimator.

    Forward:  returns x.round()   (exact rounding — no approximation)
    Backward: returns grad_output  (identity — gradient passes through unchanged)

    Usage
    -----
    >>> x = torch.tensor([0.3, 0.7, 1.2, -0.6], requires_grad=True)
    >>> y = StraightThroughRound.apply(x)
    >>> y   # tensor([0., 1., 1., -1.])
    >>> y.sum().backward()
    >>> x.grad  # tensor([1., 1., 1., 1.])  — all ones, not zeros!
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Round x to nearest integer. Save x for backward if needed.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context object for saving tensors and non-tensor data.
        x : torch.Tensor
            Input tensor (any shape, any dtype).

        Returns
        -------
        torch.Tensor
            x rounded to nearest integer. Same shape and dtype as x.

        Notes
        -----
        You do NOT need to save x for this backward pass — the STE backward
        does not use the saved input. But if you want to, you can call
        ctx.save_for_backward(x).
        """
        ###########################################################
        # YOUR CODE HERE - 2-3 lines                              #
        #                                                         #
        # Hint: just return x.round(). Optionally save x with    #
        # ctx.save_for_backward(x) for inspection.               #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        """Pass the incoming gradient through unchanged (identity).

        The Straight-Through Estimator approximates:
            d/dx[round(x)] ≈ 1

        So the backward simply returns grad_output as-is — the gradient
        flows "straight through" the rounding operation.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Context from forward (any saved tensors accessible here).
        grad_output : torch.Tensor
            Incoming gradient from the next layer (same shape as forward output).

        Returns
        -------
        torch.Tensor
            Outgoing gradient to the previous layer. Must be same shape as
            the forward input x.
        """
        ###########################################################
        # YOUR CODE HERE - 1-2 lines                              #
        #                                                         #
        # Hint: the STE backward is literally just returning the  #
        # incoming gradient unchanged. One line suffices.         #
        ###########################################################
        raise NotImplementedError("YOUR CODE HERE")
        ###########################################################


def ste_round(x: torch.Tensor) -> torch.Tensor:
    """Functional wrapper for StraightThroughRound.apply(x).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.

    Returns
    -------
    torch.Tensor
        x.round() with STE gradient.
    """
    return StraightThroughRound.apply(x)


# ---------------------------------------------------------------------------
# Implementation 2: inline detach trick
# ---------------------------------------------------------------------------

def ste_round_inline(x: torch.Tensor) -> torch.Tensor:
    """Straight-Through Estimator via the inline detach trick.

    This one-liner produces the SAME result as StraightThroughRound.apply(x):
      - Forward value:  x.round()
      - Backward grad:  identity (grad passes through unchanged)

    The trick works because:
      - (x.round() - x).detach() is a constant w.r.t. gradients
      - Adding x at the end makes d/dx(...) = 1
      - The .detach() prevents the round() and -x from entering the backward graph

    Parameters
    ----------
    x : torch.Tensor
        Input tensor (must have requires_grad=True for gradient to flow).

    Returns
    -------
    torch.Tensor
        x.round() with STE gradient (identical to ste_round(x)).

    Example
    -------
    >>> x = torch.tensor([0.73], requires_grad=True)
    >>> y = ste_round_inline(x)
    >>> y.item()   # 1.0 — rounded value
    1.0
    >>> y.backward()
    >>> x.grad     # 1.0 — identity gradient
    tensor([1.])
    """
    ###########################################################
    # YOUR CODE HERE - 1-2 lines                              #
    #                                                         #
    # Hint: return (x.round() - x).detach() + x              #
    # That's literally it. But make sure you understand WHY   #
    # this has the forward value of x.round() but gradient 1  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# Gradient verification
# ---------------------------------------------------------------------------

def verify_gradient_flow(x: torch.Tensor) -> dict:
    """Verify that gradients flow correctly through ste_round.

    Builds a simple computation graph:
        x → ste_round(x) → loss = ste_round(x).sum()

    Then calls backward() and checks that x.grad is nonzero and equals
    a tensor of all ones (since d/dx[sum(round(x))] = 1 for all elements
    via the STE identity).

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with requires_grad=True. Shape: (N,) for any N >= 1.
        Values should be non-integers (e.g., 0.3, 0.7) to make rounding
        non-trivial.

    Returns
    -------
    dict with keys:
        "forward_correct" : bool
            True if ste_round(x) matches x.round() exactly.
        "grad_is_nonzero" : bool
            True if x.grad is not all zeros after backward.
        "grad_is_identity" : bool
            True if x.grad is all ones (the identity pass).
        "forward_values" : torch.Tensor
            The actual rounded output.
        "grad_values" : torch.Tensor
            The actual gradient at x.

    Notes
    -----
    Make sure to clone x and set requires_grad=True inside this function
    so that multiple calls don't accumulate gradients.
    """
    ###########################################################
    # YOUR CODE HERE - 8-12 lines                             #
    #                                                         #
    # Step 1: create a fresh copy of x with requires_grad=True#
    #   x_leaf = x.clone().detach().requires_grad_(True)      #
    #                                                         #
    # Step 2: apply ste_round to get y                        #
    #                                                         #
    # Step 3: compute loss = y.sum()                          #
    #                                                         #
    # Step 4: loss.backward()                                 #
    #                                                         #
    # Step 5: check results:                                  #
    #   - forward_correct: torch.allclose(y, x_leaf.round())  #
    #   - grad_is_nonzero: x_leaf.grad.abs().sum() > 0        #
    #   - grad_is_identity: torch.allclose(x_leaf.grad,       #
    #                          torch.ones_like(x_leaf))       #
    #                                                         #
    # Step 6: return the dict                                  #
    ###########################################################
    raise NotImplementedError("YOUR CODE HERE")
    ###########################################################


# ---------------------------------------------------------------------------
# Main: run all verification tests (DO NOT MODIFY)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("Exercise 2: Implement and Verify the STE")
    print("=" * 60)
    print()

    # Test vectors: non-integer values so rounding is non-trivial
    test_inputs = [
        torch.tensor([0.3, 0.7, -0.3, -0.7, 1.4, -1.4, 2.6, -2.6]),
        torch.tensor([0.49, 0.51, -0.49, -0.51]),
        torch.randn(16),
    ]

    # -----------------------------------------------------------------------
    # Test 1: Forward correctness — output must equal x.round()
    # -----------------------------------------------------------------------
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

    # -----------------------------------------------------------------------
    # Test 2: Gradient flow — gradients must be nonzero (identity pass)
    # -----------------------------------------------------------------------
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

    assert result["forward_correct"],  "Forward output does not match x.round()"
    assert result["grad_is_nonzero"],  "Gradient is zero — STE not working!"
    assert result["grad_is_identity"], "Gradient is not identity — STE implementation wrong!"
    print("  ✓ Gradient flows correctly as identity (STE working)")
    print()

    # -----------------------------------------------------------------------
    # Test 3: Both implementations give identical results (forward + grad)
    # -----------------------------------------------------------------------
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
        assert values_match, f"Forward values differ for input {i+1}!"
        assert grads_match,  f"Gradients differ for input {i+1}!"

    print("  ✓ Both implementations produce identical forward values AND gradients")
    print()

    # -----------------------------------------------------------------------
    # Test 4: Numerical gradient check (optional advanced verification)
    # -----------------------------------------------------------------------
    print("TEST 4: Numerical gradient verification")
    print("-" * 40)

    # Use a non-standard weighting loss to avoid trivial cases
    x_num = torch.tensor([0.3, 0.8, 1.6, -0.4], dtype=torch.float64)
    x_num.requires_grad_(True)

    # Compute analytical gradient via STE
    y = ste_round(x_num)
    loss = (y * torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)).sum()
    loss.backward()
    analytical_grad = x_num.grad.clone()

    # The STE says gradient = incoming gradient (weights [1,2,3,4])
    expected_grad = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    numerical_match = torch.allclose(analytical_grad, expected_grad)
    print(f"  Analytical STE grad: {analytical_grad.tolist()}")
    print(f"  Expected (weights):  {expected_grad.tolist()}")
    print(f"  Match: {numerical_match}")
    assert numerical_match, "STE gradient doesn't match expected (incoming gradient)"
    print("  ✓ STE gradient equals incoming gradient (identity pass confirmed)")
    print()

    print("=" * 60)
    print("ALL STE TESTS PASSED")
    print("=" * 60)
    print()
    print("Both STE implementations are correct and identical.")
    print("autograd.Function: explicit forward/backward, pedagogically clear")
    print("inline trick:      (x.round()-x).detach()+x — used in the paper")
