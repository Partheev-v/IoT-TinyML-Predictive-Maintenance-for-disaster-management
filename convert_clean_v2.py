import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/fan_fault_model_v2_clean.keras"
TFLITE_PATH = "model/fan_fault_model_v2.tflite"

print("=" * 60)
print(" CLEAN MODEL V2 -> TFLITE")
print("=" * 60)

print("TensorFlow:", tf.__version__)

print("\nLoading clean model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

model.summary()

# ------------------------------------------------------------
# Convert directly from Keras
# ------------------------------------------------------------

print("\nConverting to Float32 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

converter.optimizations = []

tflite_model = converter.convert()

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

with open(
    TFLITE_PATH,
    "wb"
) as f:
    f.write(tflite_model)

print("\nConversion SUCCESSFUL!")

print("Saved:")
print(TFLITE_PATH)

print("Size:")
print(len(tflite_model), "bytes")

# ------------------------------------------------------------
# Verify
# ------------------------------------------------------------

print("\nLoading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput:")
print(input_details)

print("\nOutput:")
print(output_details)

# ------------------------------------------------------------
# Test inference
# ------------------------------------------------------------

test_input = np.array([
    [
        24.0,
        90.0,
        20.0,
        3.5,
        40.0
    ]
], dtype=np.float32)

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

interpreter.set_tensor(
    input_index,
    test_input
)

interpreter.invoke()

prediction = interpreter.get_tensor(
    output_index
)

print("\nTest prediction:")
print(prediction)

print("\nPredicted class:")
print(np.argmax(prediction))

print("\n============================================================")
print(" TFLITE V2 READY")
print("============================================================")
