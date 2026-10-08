import os
import numpy as np
import tensorflow as tf

# ============================================================
# TinyML FAN - SAFE MODEL V2 TFLITE CONVERSION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

KERAS_MODEL = os.path.join(
    BASE_DIR,
    "model",
    "fan_fault_model_v2.keras"
)

TFLITE_MODEL = os.path.join(
    BASE_DIR,
    "model",
    "fan_fault_model_v2.tflite"
)

print()
print("=" * 60)
print(" TINYML FAN - SAFE MODEL V2 TFLITE CONVERSION")
print("=" * 60)

print()
print("TensorFlow:", tf.__version__)

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Model V2...")

model = tf.keras.models.load_model(
    KERAS_MODEL,
    compile=False
)

print("Model loaded successfully.")

print()
model.summary()

# ============================================================
# CREATE CONCRETE FUNCTION
# ============================================================

print()
print("=" * 60)
print(" CREATING INFERENCE FUNCTION")
print("=" * 60)

@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=[1, 5],
            dtype=tf.float32,
            name="sensor_features"
        )
    ]
)
def inference(x):

    return model(x, training=False)

concrete_function = (
    inference.get_concrete_function()
)

print()
print("Concrete function created.")

# ============================================================
# CONVERT
# ============================================================

print()
print("=" * 60)
print(" CONVERTING TO FLOAT32 TFLITE")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [concrete_function],
    model
)

# IMPORTANT:
# Keep conversion completely unoptimized.
converter.optimizations = []

print()
print("Starting conversion...")

try:

    tflite_model = converter.convert()

except Exception as e:

    print()
    print("CONVERSION FAILED")
    print()
    print(repr(e))

    raise SystemExit(1)

# ============================================================
# SAVE
# ============================================================

with open(
    TFLITE_MODEL,
    "wb"
) as f:

    f.write(tflite_model)

print()
print("=" * 60)
print(" CONVERSION SUCCESSFUL")
print("=" * 60)

print()
print("TFLite model:")
print(TFLITE_MODEL)

print()
print("Size:")
print(len(tflite_model), "bytes")

# ============================================================
# VERIFY TFLITE
# ============================================================

print()
print("=" * 60)
print(" VERIFYING TFLITE MODEL")
print("=" * 60)

try:

    interpreter = tf.lite.Interpreter(
        model_path=TFLITE_MODEL
    )

    interpreter.allocate_tensors()

except Exception as e:

    print()
    print("TFLITE VERIFICATION FAILED")
    print(repr(e))

    raise SystemExit(1)

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)

print()
print("Input details:")
print(input_details)

print()
print("Output details:")
print(output_details)

# ============================================================
# TEST ONE SAMPLE
# ============================================================

test_input = np.zeros(
    (1, 5),
    dtype=np.float32
)

input_index = (
    input_details[0]["index"]
)

output_index = (
    output_details[0]["index"]
)

interpreter.set_tensor(
    input_index,
    test_input
)

interpreter.invoke()

output = interpreter.get_tensor(
    output_index
)

print()
print("Test output:")
print(output)

print()
print("Output shape:")
print(output.shape)

# ============================================================
# FINAL
# ============================================================

if output.shape == (1, 3):

    print()
    print("=" * 60)
    print(" PASS")
    print("=" * 60)

    print()
    print("Model V2 successfully converted to TFLite.")
    print("Input:  5 features")
    print("Output: 3 classes")

else:

    print()
    print("=" * 60)
    print(" ERROR")
    print("=" * 60)

    print()
    print("Unexpected output shape.")

print()
