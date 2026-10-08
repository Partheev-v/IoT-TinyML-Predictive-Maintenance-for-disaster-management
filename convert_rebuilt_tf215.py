import os
import numpy as np
import tensorflow as tf

MODEL_PATH = "model/fan_fault_model_v2_tf215.h5"
TFLITE_PATH = "model/fan_fault_model_v2.tflite"

print("=" * 60)
print(" CONVERTING REBUILT V2 MODEL TO TFLITE")
print("=" * 60)

print("TensorFlow:", tf.__version__)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("\nModel loaded.")
model.summary()

# ============================================================
# CONVERT
# ============================================================

print("\nConverting to Float32 TFLite...")

converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)

tflite_model = converter.convert()

# ============================================================
# SAVE
# ============================================================

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("\nSUCCESS!")
print("TFLite model:")
print(TFLITE_PATH)

print("Size:")
print(len(tflite_model), "bytes")

# ============================================================
# VERIFY
# ============================================================

print("\nVerifying TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nInput shape:")
print(input_details[0]["shape"])

print("Input type:")
print(input_details[0]["dtype"])

print("\nOutput shape:")
print(output_details[0]["shape"])

print("Output type:")
print(output_details[0]["dtype"])

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

interpreter.set_tensor(
    input_details[0]["index"],
    test_input
)

interpreter.invoke()

prediction = interpreter.get_tensor(
    output_details[0]["index"]
)

print("\nTFLite prediction:")
print(prediction)

print("\nPredicted class:")
print(int(np.argmax(prediction)))

print("\n" + "=" * 60)
print(" TFLITE CONVERSION COMPLETE")
print("=" * 60)
