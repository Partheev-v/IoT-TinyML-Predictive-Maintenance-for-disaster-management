import os
import numpy as np
import tensorflow as tf

SOURCE_MODEL = "model/fan_fault_model_v2.keras"
OUTPUT_MODEL = "model/fan_fault_model_v2_clean.keras"

print("=" * 60)
print(" REBUILDING CLEAN MODEL V2")
print("=" * 60)

print("TensorFlow:", tf.__version__)

# ============================================================
# LOAD ORIGINAL MODEL
# ============================================================

print("\nLoading original V2 model...")

old_model = tf.keras.models.load_model(
    SOURCE_MODEL,
    compile=False
)

print("Original model loaded.")

old_model.summary()

# ============================================================
# CREATE EXACT SAME ARCHITECTURE
# ============================================================

print("\nCreating clean inference model...")

clean_model = tf.keras.Sequential([
    tf.keras.Input(
        shape=(5,),
        name="sensor_features"
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu",
        name="dense"
    ),

    tf.keras.layers.Dropout(
        0.2,
        name="dropout"
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu",
        name="dense_1"
    ),

    tf.keras.layers.Dropout(
        0.2,
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

print("\nClean model created.")

clean_model.summary()

# ============================================================
# COPY WEIGHTS BY LAYER NAME
# ============================================================

print("\nCopying trained weights...")

for old_layer in old_model.layers:

    if not old_layer.weights:
        continue

    try:
        new_layer = clean_model.get_layer(
            old_layer.name
        )

        new_layer.set_weights(
            old_layer.get_weights()
        )

        print(
            "Copied:",
            old_layer.name
        )

    except Exception as e:

        print(
            "ERROR copying:",
            old_layer.name
        )

        print(e)

        raise

print("\nAll trained weights copied successfully.")

# ============================================================
# SAVE
# ============================================================

print("\nSaving clean model...")

clean_model.save(
    OUTPUT_MODEL
)

print("\nClean model saved:")
print(OUTPUT_MODEL)

# ============================================================
# VERIFY WEIGHTS / PREDICTIONS
# ============================================================

print("\n============================================================")
print(" VERIFYING ORIGINAL VS CLEAN MODEL")
print("============================================================")

test_input = np.array([
    [
        24.0,
        90.0,
        20.0,
        3.5,
        40.0
    ]
], dtype=np.float32)

old_prediction = old_model.predict(
    test_input,
    verbose=0
)

clean_prediction = clean_model.predict(
    test_input,
    verbose=0
)

print("\nOriginal model prediction:")
print(old_prediction)

print("\nClean model prediction:")
print(clean_prediction)

difference = np.max(
    np.abs(
        old_prediction -
        clean_prediction
    )
)

print("\nMaximum prediction difference:")
print(difference)

old_class = int(
    np.argmax(old_prediction)
)

clean_class = int(
    np.argmax(clean_prediction)
)

print("\nOriginal predicted class:")
print(old_class)

print("Clean predicted class:")
print(clean_class)

# ============================================================
# FINAL CHECK
# ============================================================

print("\n============================================================")
print(" VERIFICATION RESULT")
print("============================================================")

if difference < 0.000001 and old_class == clean_class:

    print("PASS")
    print("Clean model matches the original model.")

else:

    print("WARNING")
    print("Clean model does not exactly match original model.")

print("\n============================================================")
