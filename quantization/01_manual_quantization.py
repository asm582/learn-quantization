"""
Step 1: Quantization from scratch, no libraries, just numpy.

The core idea of quantization: neural network weights are normally stored as
32-bit floats (fp32). Each number takes 4 bytes. Quantization stores each
number in fewer bits (e.g. 8-bit integers = 1 byte), trading a little
precision for a lot less memory and faster math.

We'll implement "absmax" symmetric int8 quantization -- the same basic idea
used inside bitsandbytes, GPTQ, etc, just without the engineering around it.
"""

import numpy as np

np.random.seed(0)

# ---------------------------------------------------------------------------
# 1. Pretend this is a small weight matrix from a neural network layer.
#    Real weights are roughly bell-curve distributed around 0, so we simulate that.
# ---------------------------------------------------------------------------
weights_fp32 = np.random.randn(4, 8).astype(np.float32) * 0.5
print("Original fp32 weights:")
print(weights_fp32)
print(f"dtype={weights_fp32.dtype}, bytes used = {weights_fp32.nbytes}")

# ---------------------------------------------------------------------------
# 2. Quantize to int8.
#
#    int8 can represent whole numbers from -127 to 127 (we avoid -128 to keep
#    things symmetric around zero -- this is what "symmetric" quantization means).
#
#    We need a "scale" factor that maps our float range onto that int range.
#    scale = max(abs(weights)) / 127
#
#    Then: quantized_value = round(original_value / scale)
# ---------------------------------------------------------------------------
def quantize_int8(x: np.ndarray):
    scale = np.abs(x).max() / 127.0
    q = np.round(x / scale).astype(np.int8)
    return q, scale


def dequantize_int8(q: np.ndarray, scale: float):
    return q.astype(np.float32) * scale


weights_int8, scale = quantize_int8(weights_fp32)
print("\nQuantized int8 weights:")
print(weights_int8)
print(f"scale factor = {scale:.6f}")
print(f"dtype={weights_int8.dtype}, bytes used = {weights_int8.nbytes}")
print(f"Memory saved: {weights_fp32.nbytes / weights_int8.nbytes:.1f}x smaller")

# ---------------------------------------------------------------------------
# 3. Dequantize back to float to see how much information we lost.
#    We can't get the exact original numbers back -- that's the "lossy" part.
# ---------------------------------------------------------------------------
weights_dequantized = dequantize_int8(weights_int8, scale)
print("\nDequantized back to float (approximation of the original):")
print(weights_dequantized)

error = np.abs(weights_fp32 - weights_dequantized)
print(f"\nMax absolute error: {error.max():.6f}")
print(f"Mean absolute error: {error.mean():.6f}")
print(f"Original value range: [{weights_fp32.min():.4f}, {weights_fp32.max():.4f}]")

# ---------------------------------------------------------------------------
# 4. Why does a single outlier ruin everything?
#    absmax scaling uses the single largest magnitude value to set the scale
#    for the WHOLE tensor. One outlier stretches the scale, wasting int8's
#    range on the other 99% of values. This is exactly the problem that
#    real quantization schemes (per-channel scaling, LLM.int8(), GPTQ, AWQ)
#    are designed to fix.
# ---------------------------------------------------------------------------
print("\n--- Now let's add one outlier value and see what happens ---")
weights_with_outlier = weights_fp32.copy()
weights_with_outlier[0, 0] = 10.0  # a big outlier

q_outlier, scale_outlier = quantize_int8(weights_with_outlier)
dq_outlier = dequantize_int8(q_outlier, scale_outlier)
error_outlier = np.abs(weights_with_outlier - dq_outlier)

print(f"New scale factor: {scale_outlier:.6f} (was {scale:.6f} without the outlier)")
print(f"Mean absolute error on the OTHER 31 values (excluding outlier): "
      f"{np.delete(error_outlier, 0):.6f}" if False else
      f"Mean absolute error on the other values: {np.mean(np.delete(error_outlier.flatten(), 0)):.6f}")
print("^ Notice this error is much bigger than before, even though those values didn't change.")
print("  A single large-magnitude weight forced the scale to stretch, wasting precision on everything else.")
