import os
import numpy as np
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/fan_fault_model_v2_clean.keras"
TFLITE_PATH = "model/fan_fault_model_v2.tflite"

print("=" * 60)
print(" TINYML FAN - TF 2.15 TFLITE CONVERSION")
print("=" * 60)

print("\nTensorFlow:", tf.__version__)

# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("\nERROR: Clean model not found:")
    print(MODEL_PATH)
    raise SystemExit(1)

print("\nLoading clean model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Clean model loaded successfully.")

print("\nModel summary:")
model.summary()

# ============================================================
# TEST KERAS MODEL
# ============================================================

print("\n" + "=" * 60)
print(" TESTING KERAS MODEL")
print("=" * 60)

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

keras_prediction = model.predict(
    test_input,
    verbose=0
)

print("\nTest input:")
print(test_input)

print("\nKeras prediction:")
print(keras_prediction)

print("\nKeras class:")
print(int(np.argmax(keras_prediction)))

# ============================================================
# CONVERT
# ============================================================

print("\n" + "=" * 60)
print(" CONVERTING TO FLOAT32 TFLITE")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = []

print("\nStarting conversion...")

tflite_model = converter.convert()

print("\nTFLite conversion SUCCESSFUL!")

# ============================================================
# SAVE
# ============================================================

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print("\nSaved:")
print(TFLITE_PATH)

print("Size:")
print(len(tflite_model), "bytes")

# ============================================================
# VERIFY TFLITE
# ============================================================

print("\n" + "=" * 60)
print(" VERIFYING TFLITE MODEL")
print("=" * 60)

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

# ============================================================
# TFLITE INFERENCE
# ============================================================

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

interpreter.set_tensor(
    input_index,
    test_input
)

interpreter.invoke()

tflite_prediction = interpreter.get_tensor(
    output_index
)

print("\nTFLite prediction:")
print(tflite_prediction)

print("\nTFLite class:")
print(int(np.argmax(tflite_prediction)))

# ============================================================
# COMPARE
# ============================================================

difference = np.max(
    np.abs(
        keras_prediction -
        tflite_prediction
    )
)

print("\nMaximum prediction difference:")
print(difference)

print("\n" + "=" * 60)
print(" FINAL RESULT")
print("=" * 60)

if difference < 0.001:

    print("PASS")
    print("Keras and TFLite predictions match.")

else:

    print("WARNING")
    print("Keras and TFLite predictions differ.")

print("=" * 60)
