import os
import numpy as np
import tensorflow as tf

WEIGHTS_PATH = "model/v2_weights.npz"
OUTPUT_MODEL = "model/fan_fault_model_v2_tf215.h5"

print("=" * 60)
print(" REBUILDING TINYML FAN V2 FOR TF 2.15")
print("=" * 60)

print("TensorFlow:", tf.__version__)

# ============================================================
# LOAD WEIGHTS
# ============================================================

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(
        f"Missing weights: {WEIGHTS_PATH}"
    )

data = np.load(WEIGHTS_PATH)

# ============================================================
# CREATE TF 2.15 COMPATIBLE MODEL
# ============================================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(
        shape=(5,),
        name="sensor_features"
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu",
        name="dense"
    ),

    tf.keras.layers.Dropout(
        0.0,
        name="dropout"
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_1"
    ),

    tf.keras.layers.Dropout(
        0.0,
        name="dropout_1"
    ),

    tf.keras.layers.Dense(
        8,
        activation="relu",
        name="dense_2"
    ),

    tf.keras.layers.Dense(
        3,
        activation="softmax",
        name="dense_3"
    )
])

print("\nModel created.")

# ============================================================
# SET TRAINED WEIGHTS
# ============================================================

layer_names = [
    "dense",
    "dense_1",
    "dense_2",
    "dense_3"
]

for i, name in enumerate(layer_names):

    layer = model.get_layer(name)

    layer.set_weights([
        data[f"W{i}"],
        data[f"b{i}"]
    ])

    print(
        f"Copied weights: {name}"
    )

# ============================================================
# VERIFY
# ============================================================

model.build((None, 5))

print("\nModel summary:")
model.summary()

# ============================================================
# TEST
# ============================================================

test_input = np.array(
    [[
        25.0,
        50.0,
        0.10,
        0.50,
        100.0
    ]],
    dtype=np.float32
)

prediction = model.predict(
    test_input,
    verbose=0
)

print("\nTest prediction:")
print(prediction)

print("\nPredicted class:")
print(int(np.argmax(prediction)))

# ============================================================
# SAVE H5
# ============================================================

model.save(
    OUTPUT_MODEL,
    include_optimizer=False
)

print("\nModel saved:")
print(OUTPUT_MODEL)

print("\n" + "=" * 60)
print(" REBUILD COMPLETE")
print("=" * 60)

