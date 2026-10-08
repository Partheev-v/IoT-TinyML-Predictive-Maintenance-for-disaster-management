import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# TINYML FAN - MODEL V2 TFLITE CONVERSION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

KERAS_MODEL = os.path.join(
    MODEL_DIR,
    "fan_fault_model_v2.keras"
)

SCALER_FILE = os.path.join(
    MODEL_DIR,
    "scaler_v2.pkl"
)

TFLITE_MODEL = os.path.join(
    MODEL_DIR,
    "fan_fault_model_v2.tflite"
)

DATA_FILE = os.path.join(
    DATA_DIR,
    "fan_data_combined.csv"
)

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
# HEADER
# ============================================================

print()
print("=" * 60)
print(" TINYML FAN - MODEL V2 TFLITE CONVERSION")
print("=" * 60)

print()
print("TensorFlow:", tf.__version__)

# ============================================================
# CHECK FILES
# ============================================================

for path in [
    KERAS_MODEL,
    SCALER_FILE,
    DATA_FILE
]:

    if not os.path.exists(path):

        print()
        print("ERROR: File not found:")
        print(path)

        raise SystemExit(1)

# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading Model V2...")

model = tf.keras.models.load_model(
    KERAS_MODEL,
    compile=False
)

print("Model V2 loaded successfully.")

print()
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ============================================================
# LOAD DATA
# ============================================================

print()
print("Loading combined dataset...")

df = pd.read_csv(DATA_FILE)

print("Dataset rows:", len(df))

# ============================================================
# CLEAN DATA
# ============================================================

required_columns = FEATURES + [LABEL_COLUMN]

df = df.dropna(
    subset=required_columns
).copy()

for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df[LABEL_COLUMN] = pd.to_numeric(
    df[LABEL_COLUMN],
    errors="coerce"
)

df = df.dropna(
    subset=required_columns
).copy()

df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(
    np.int32
)

# ============================================================
# LOAD SCALER
# ============================================================

print()
print("Loading V2 scaler...")

scaler = joblib.load(
    SCALER_FILE
)

print("V2 scaler loaded.")

# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[FEATURES].astype(
    np.float32
)

y = df[LABEL_COLUMN].values.astype(
    np.int32
)

# Use DataFrame so StandardScaler receives feature names
X_scaled = scaler.transform(
    X
).astype(
    np.float32
)

# ============================================================
# MODEL CHECK
# ============================================================

print()
print("=" * 60)
print(" MODEL V2 KERAS CHECK")
print("=" * 60)

keras_output = model.predict(
    X_scaled,
    verbose=0
)

keras_pred = np.argmax(
    keras_output,
    axis=1
)

keras_accuracy = accuracy_score(
    y,
    keras_pred
)

print()
print(
    f"Accuracy on combined dataset: "
    f"{keras_accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        y,
        keras_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

print("Confusion matrix:")

print(
    confusion_matrix(
        y,
        keras_pred,
        labels=[0, 1, 2]
    )
)

# ============================================================
# SAVE AS SAVEDMODEL FIRST
# ============================================================

print()
print("=" * 60)
print(" EXPORTING MODEL V2")
print("=" * 60)

SAVED_MODEL_DIR = os.path.join(
    MODEL_DIR,
    "fan_fault_model_v2_savedmodel"
)

# Remove old SavedModel directory if present
if os.path.exists(SAVED_MODEL_DIR):

    import shutil

    shutil.rmtree(
        SAVED_MODEL_DIR
    )

print()
print("Saving SavedModel...")

try:

    tf.saved_model.save(
        model,
        SAVED_MODEL_DIR
    )

except Exception as e:

    print()
    print("ERROR while saving SavedModel:")
    print(repr(e))

    raise SystemExit(1)

print("SavedModel created:")
print(SAVED_MODEL_DIR)

# ============================================================
# CONVERT SAVEDMODEL TO TFLITE
# ============================================================

print()
print("=" * 60)
print(" CONVERTING SAVEDMODEL TO TFLITE")
print("=" * 60)

print()
print("Starting TFLite conversion...")

try:

    converter = (
        tf.lite.TFLiteConverter
        .from_saved_model(
            SAVED_MODEL_DIR
        )
    )

    # Float32 model
    converter.optimizations = []

    tflite_model = converter.convert()

except Exception as e:

    print()
    print("ERROR: TFLite conversion failed.")
    print()
    print(repr(e))

    raise SystemExit(1)

# ============================================================
# SAVE TFLITE
# ============================================================

with open(
    TFLITE_MODEL,
    "wb"
) as f:

    f.write(
        tflite_model
    )

print()
print("TFLite conversion SUCCESSFUL!")

print()
print("Saved:")
print(TFLITE_MODEL)

print()
print(
    "Size:",
    len(tflite_model),
    "bytes"
)

# ============================================================
# LOAD TFLITE
# ============================================================

print()
print("=" * 60)
print(" LOADING TFLITE MODEL")
print("=" * 60)

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

print()
print("Input:")
print(input_details)

print()
print("Output:")
print(output_details)

# ============================================================
# TEST TFLITE
# ============================================================

print()
print("=" * 60)
print(" TESTING TFLITE MODEL")
print("=" * 60)

input_index = (
    input_details[0]["index"]
)

output_index = (
    output_details[0]["index"]
)

tflite_predictions = []

for sample in X_scaled:

    sample = np.expand_dims(
        sample,
        axis=0
    ).astype(
        np.float32
    )

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
    y,
    tflite_pred
)

print()
print(
    f"TFLite accuracy: "
    f"{tflite_accuracy * 100:.2f}%"
)

print()

print(
    classification_report(
        y,
        tflite_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )
)

print("TFLite confusion matrix:")

print(
    confusion_matrix(
        y,
        tflite_pred,
        labels=[0, 1, 2]
    )
)

# ============================================================
# KERAS VS TFLITE
# ============================================================

agreement = np.mean(
    keras_pred == tflite_pred
)

accuracy_difference = abs(
    keras_accuracy -
    tflite_accuracy
)

print()
print("=" * 60)
print(" KERAS vs TFLITE")
print("=" * 60)

print()
print(
    f"Keras accuracy:  "
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
# FINAL RESULT
# ============================================================

print()
print("=" * 60)
print(" RESULT")
print("=" * 60)

if agreement >= 0.99:

    print()
    print("PASS")

    print(
        "Keras and TFLite predictions agree."
    )

else:

    print()
    print("WARNING")

    print(
        "Keras and TFLite predictions differ."
    )

print()
print("=" * 60)
