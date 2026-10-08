import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = "data/fan_data.csv"

KERAS_MODEL = "model/fan_fault_model.keras"
TFLITE_MODEL = "model/fan_fault_model.tflite"

FEATURES = [
    "temperature",
    "humidity",
    "vibration",
    "current",
    "sound"
]

LABEL_COLUMN = "label"

CLASS_NAMES = [
    "NORMAL",
    "WARNING",
    "FAULT"
]

# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print(" TFLITE CONVERSION + TEST")
print("==============================================")

print("\nTensorFlow:", tf.__version__)

df = pd.read_csv(CSV_FILE)

df = df.dropna()

for column in FEATURES + [LABEL_COLUMN]:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna()

X = df[FEATURES].values.astype(np.float32)
y = df[LABEL_COLUMN].values.astype(np.int32)

# ============================================================
# SAME SPLIT AS TRAINING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# SAME STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)

# ============================================================
# LOAD KERAS MODEL
# ============================================================

print("\nLoading Keras model...")

model = tf.keras.models.load_model(
    KERAS_MODEL
)

print("Keras model loaded.")

# ============================================================
# VERIFY KERAS MODEL
# ============================================================

print("\n==============================================")
print(" KERAS MODEL TEST")
print("==============================================")

keras_output = model.predict(
    X_test_scaled,
    verbose=0
)

keras_pred = np.argmax(
    keras_output,
    axis=1
)

keras_accuracy = accuracy_score(
    y_test,
    keras_pred
)

print(
    f"Keras accuracy: "
    f"{keras_accuracy * 100:.2f}%"
)

print(
    classification_report(
        y_test,
        keras_pred,
        target_names=CLASS_NAMES
    )
)

print("Keras confusion matrix:")

print(
    confusion_matrix(
        y_test,
        keras_pred
    )
)

# ============================================================
# CREATE CONCRETE FUNCTION
# ============================================================

print("\n==============================================")
print(" CREATING CONCRETE FUNCTION")
print("==============================================")

input_shape = (
    1,
    len(FEATURES)
)

@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=input_shape,
            dtype=tf.float32,
            name="sensor_features"
        )
    ]
)
def serving_function(x):

    return {
        "output": model(x)
    }

concrete_function = (
    serving_function
    .get_concrete_function()
)

print("Concrete function created.")

# ============================================================
# CONVERT USING CONCRETE FUNCTION
# ============================================================

print("\n==============================================")
print(" CONVERTING TO TFLITE")
print("==============================================")

converter = (
    tf.lite.TFLiteConverter
    .from_concrete_functions(
        [concrete_function]
    )
)

# IMPORTANT:
# No optimization yet.
#
# We first need a working Float32 TFLite model.

tflite_model = converter.convert()

# ============================================================
# SAVE TFLITE
# ============================================================

with open(
    TFLITE_MODEL,
    "wb"
) as f:

    f.write(tflite_model)

print("\nTFLite conversion SUCCESSFUL!")

print(
    "Saved:",
    TFLITE_MODEL
)

print(
    "Size:",
    len(tflite_model),
    "bytes"
)

# ============================================================
# LOAD TFLITE
# ============================================================

print("\n==============================================")
print(" LOADING TFLITE MODEL")
print("==============================================")

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL
)

interpreter.allocate_tensors()

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)

print("\nInput:")
print(input_details)

print("\nOutput:")
print(output_details)

# ============================================================
# TEST TFLITE
# ============================================================

print("\n==============================================")
print(" TESTING TFLITE MODEL")
print("==============================================")

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

tflite_predictions = []

for sample in X_test_scaled:

    sample = np.expand_dims(
        sample,
        axis=0
    ).astype(np.float32)

    interpreter.set_tensor(
        input_index,
        sample
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_index
    )

    tflite_predictions.append(
        output[0]
    )

tflite_predictions = np.array(
    tflite_predictions
)

tflite_pred = np.argmax(
    tflite_predictions,
    axis=1
)

# ============================================================
# TFLITE RESULTS
# ============================================================

tflite_accuracy = accuracy_score(
    y_test,
    tflite_pred
)

print(
    f"\nTFLite accuracy: "
    f"{tflite_accuracy * 100:.2f}%"
)

print("\nTFLite classification report:")

print(
    classification_report(
        y_test,
        tflite_pred,
        target_names=CLASS_NAMES
    )
)

print("\nTFLite confusion matrix:")

print(
    confusion_matrix(
        y_test,
        tflite_pred
    )
)

# ============================================================
# COMPARE KERAS AND TFLITE
# ============================================================

agreement = np.mean(
    keras_pred == tflite_pred
)

accuracy_difference = abs(
    keras_accuracy -
    tflite_accuracy
)

print("\n==============================================")
print(" KERAS vs TFLITE")
print("==============================================")

print(
    f"\nKeras accuracy:  "
    f"{keras_accuracy * 100:.2f}%"
)

print(
    f"TFLite accuracy: "
    f"{tflite_accuracy * 100:.2f}%"
)

print(
    f"Prediction agreement: "
    f"{agreement * 100:.2f}%"
)

print(
    f"Accuracy difference: "
    f"{accuracy_difference * 100:.4f}%"
)

# ============================================================
# FINAL STATUS
# ============================================================

print("\n==============================================")
print(" RESULT")
print("==============================================")

if agreement >= 0.99:

    print("PASS")
    print(
        "Keras and TFLite predictions agree."
    )

else:

    print("WARNING")
    print(
        "Keras and TFLite predictions differ."
    )

print("\n==============================================")