import numpy as np
import tensorflow as tf

MODEL_PATH = "model/fan_fault_model_v2_clean.keras"

print("=" * 60)
print(" EXPORTING TINYML FAN V2 WEIGHTS")
print("=" * 60)

print("TensorFlow:", tf.__version__)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded successfully.")
model.summary()

# ------------------------------------------------------------
# Extract Dense layer weights
# ------------------------------------------------------------

dense_layers = [
    layer for layer in model.layers
    if isinstance(layer, tf.keras.layers.Dense)
]

print("\nDense layers found:", len(dense_layers))

weights = {}

for i, layer in enumerate(dense_layers):

    w, b = layer.get_weights()

    print(
        f"\nLayer {i}: {layer.name}"
    )

    print("Weights shape:", w.shape)
    print("Bias shape:", b.shape)

    weights[f"W{i}"] = w.astype(np.float32)
    weights[f"b{i}"] = b.astype(np.float32)

# ------------------------------------------------------------
# Save weights
# ------------------------------------------------------------

output = "model/v2_weights.npz"

np.savez(
    output,
    **weights
)

print("\nWeights exported successfully:")
print(output)

print("\n" + "=" * 60)
print(" EXPORT COMPLETE")
print("=" * 60)
