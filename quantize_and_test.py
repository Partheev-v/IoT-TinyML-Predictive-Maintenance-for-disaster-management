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

FLOAT_TFLITE_MODEL = "model/fan_fault_model.tflite"

INT8_TFLITE_MODEL = "model/fan_fault_model_int8.tflite"

MODEL_HEADER = "model/model_data.h"

NORMALIZATION_HEADER = "model/normalization_data.h"

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
# START
# ============================================================

print("\n==============================================")
print(" FAN TinyML INT8 QUANTIZATION")
print("==============================================")

print("\nTensorFlow version:")
print(tf.__version__)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

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

print("Dataset shape:", df.shape)

# ============================================================
# TRAIN / TEST SPLIT
# SAME SPLIT USED DURING TRAINING
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)

print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))

# ============================================================
# NORMALIZATION
# SAME STANDARDIZATION USED DURING TRAINING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
).astype(np.float32)

X_test_scaled = scaler.transform(
    X_test
).astype(np.float32)

scaler_mean = scaler.mean_

scaler_scale = scaler.scale_

# ============================================================
# LOAD KERAS MODEL
# ============================================================

print("\nLoading Keras model...")

model = tf.keras.models.load_model(
    KERAS_MODEL
)

print("Keras model loaded.")

# ============================================================
# BASELINE KERAS TEST
# ============================================================

print("\n==============================================")
print(" BASELINE KERAS MODEL")
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
    f"\nKeras accuracy: "
    f"{keras_accuracy * 100:.2f}%"
)

print("\nKeras classification report:")

print(
    classification_report(
        y_test,
        keras_pred,
        target_names=CLASS_NAMES
    )
)

# ============================================================
# CREATE CONCRETE FUNCTION
# ============================================================

@tf.function(
    input_signature=[
        tf.TensorSpec(
            shape=(1, 5),
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

# ============================================================
# INT8 REPRESENTATIVE DATASET
# ============================================================

print("\n==============================================")
print(" CREATING REPRESENTATIVE DATASET")
print("==============================================")

def representative_dataset():

    # Use training data only.
    #
    # This is important:
    # Test data must NOT be used for calibration.

    for sample in X_train_scaled:

        sample = np.expand_dims(
            sample,
            axis=0
        ).astype(np.float32)

        yield [sample]


print(
    "Representative samples:",
    len(X_train_scaled)
)

# ============================================================
# CREATE INT8 CONVERTER
# ============================================================

print("\n==============================================")
print(" CONVERTING TO INT8 TFLITE")
print("==============================================")

converter = (
    tf.lite.TFLiteConverter
    .from_concrete_functions(
        [concrete_function]
    )
)

# Enable optimization

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

# Representative dataset

converter.representative_dataset = (
    representative_dataset
)

# Force full INT8 quantization

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# ESP32 will provide FLOAT32 sensor input.
#
# Therefore keep the model input FLOAT32.
#
# TFLite will internally quantize the input.

converter.inference_input_type = tf.float32

# Keep FLOAT32 output so ESP32 can easily read
# the three class probabilities.

converter.inference_output_type = tf.float32

# ============================================================
# CONVERT
# ============================================================

int8_model = converter.convert()

# ============================================================
# SAVE INT8 MODEL
# ============================================================

with open(
    INT8_TFLITE_MODEL,
    "wb"
) as f:

    f.write(int8_model)

print("\nINT8 conversion successful!")

print(
    "Saved:",
    INT8_TFLITE_MODEL
)

print(
    "INT8 model size:",
    len(int8_model),
    "bytes"
)

# ============================================================
# FLOAT MODEL SIZE
# ============================================================

if os.path.exists(
    FLOAT_TFLITE_MODEL
):

    float_size = os.path.getsize(
        FLOAT_TFLITE_MODEL
    )

    int8_size = len(int8_model)

    reduction = (
        1 -
        (int8_size / float_size)
    ) * 100

    print(
        "\nFloat32 model size:",
        float_size,
        "bytes"
    )

    print(
        "INT8 model size:",
        int8_size,
        "bytes"
    )

    print(
        f"Size reduction: {reduction:.2f}%"
    )

# ============================================================
# LOAD INT8 TFLITE MODEL
# ============================================================

print("\n==============================================")
print(" TESTING INT8 TFLITE MODEL")
print("==============================================")

interpreter = tf.lite.Interpreter(
    model_path=INT8_TFLITE_MODEL
)

interpreter.allocate_tensors()

input_details = (
    interpreter.get_input_details()
)

output_details = (
    interpreter.get_output_details()
)

print("\nInput details:")

print(input_details)

print("\nOutput details:")

print(output_details)

# ============================================================
# VERIFY INPUT / OUTPUT TYPES
# ============================================================

print("\nInput type:")

print(
    input_details[0]["dtype"]
)

print("\nOutput type:")

print(
    output_details[0]["dtype"]
)

# ============================================================
# RUN INT8 MODEL
# ============================================================

input_index = (
    input_details[0]["index"]
)

output_index = (
    output_details[0]["index"]
)

int8_predictions = []

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

    int8_predictions.append(
        output[0]
    )

int8_predictions = np.array(
    int8_predictions
)

# ============================================================
# INT8 PREDICTIONS
# ============================================================

int8_pred = np.argmax(
    int8_predictions,
    axis=1
)

int8_accuracy = accuracy_score(
    y_test,
    int8_pred
)

# ============================================================
# INT8 RESULTS
# ============================================================

print("\n==============================================")
print(" INT8 RESULTS")
print("==============================================")

print(
    f"\nINT8 accuracy: "
    f"{int8_accuracy * 100:.2f}%"
)

print("\nINT8 classification report:")

print(
    classification_report(
        y_test,
        int8_pred,
        target_names=CLASS_NAMES
    )
)

print("\nINT8 confusion matrix:")

print(
    confusion_matrix(
        y_test,
        int8_pred
    )
)

# ============================================================
# COMPARE KERAS VS INT8
# ============================================================

agreement = np.mean(
    keras_pred == int8_pred
)

accuracy_difference = abs(
    keras_accuracy -
    int8_accuracy
)

print("\n==============================================")
print(" KERAS vs INT8")
print("==============================================")

print(
    f"\nKeras accuracy: "
    f"{keras_accuracy * 100:.2f}%"
)

print(
    f"INT8 accuracy: "
    f"{int8_accuracy * 100:.2f}%"
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
# GENERATE model_data.h
# ============================================================

print("\n==============================================")
print(" GENERATING model_data.h")
print("==============================================")

with open(
    MODEL_HEADER,
    "w"
) as f:

    f.write(
        "#ifndef MODEL_DATA_H\n"
    )

    f.write(
        "#define MODEL_DATA_H\n\n"
    )

    f.write(
        "#include <stdint.h>\n\n"
    )

    f.write(
        "// ==================================================\n"
    )

    f.write(
        "// Fan Predictive Maintenance TinyML Model\n"
    )

    f.write(
        "// INT8 Quantized TensorFlow Lite Model\n"
    )

    f.write(
        "// Automatically generated\n"
    )

    f.write(
        "// ==================================================\n\n"
    )

    f.write(
        "const unsigned char model_data[] = {\n"
    )

    for i, byte in enumerate(int8_model):

        if i % 12 == 0:

            f.write("    ")

        f.write(
            f"0x{byte:02x}"
        )

        if i != len(int8_model) - 1:

            f.write(", ")

        if i % 12 == 11:

            f.write("\n")

    f.write(
        "\n};\n\n"
    )

    f.write(
        "const unsigned int model_data_len = "
    )

    f.write(
        f"{len(int8_model)};\n\n"
    )

    f.write(
        "#endif\n"
    )

print(
    "Saved:",
    MODEL_HEADER
)

# ============================================================
# GENERATE normalization_data.h
# ============================================================

print("\n==============================================")
print(" GENERATING normalization_data.h")
print("==============================================")

with open(
    NORMALIZATION_HEADER,
    "w"
) as f:

    f.write(
        "#ifndef NORMALIZATION_DATA_H\n"
    )

    f.write(
        "#define NORMALIZATION_DATA_H\n\n"
    )

    f.write(
        "// Feature order:\n"
    )

    f.write(
        "// temperature, humidity, vibration, current, sound\n\n"
    )

    f.write(
        "const float scaler_mean[5] = {\n"
    )

    f.write(
        "    "
    )

    f.write(
        ", ".join(
            f"{value:.10f}f"
            for value in scaler_mean
        )
    )

    f.write(
        "\n};\n\n"
    )

    f.write(
        "const float scaler_scale[5] = {\n"
    )

    f.write(
        "    "
    )

    f.write(
        ", ".join(
            f"{value:.10f}f"
            for value in scaler_scale
        )
    )

    f.write(
        "\n};\n\n"
    )

    f.write(
        "#endif\n"
    )

print(
    "Saved:",
    NORMALIZATION_HEADER
)

# ============================================================
# FINAL STATUS
# ============================================================

print("\n==============================================")
print(" INT8 QUANTIZATION COMPLETE")
print("==============================================")

print("\nGenerated files:")

print(
    "1.",
    INT8_TFLITE_MODEL
)

print(
    "2.",
    MODEL_HEADER
)

print(
    "3.",
    NORMALIZATION_HEADER
)

print("\nClasses:")

for i, name in enumerate(CLASS_NAMES):

    print(
        f"{i} = {name}"
    )

print("\nFeatures:")

for i, feature in enumerate(FEATURES):

    print(
        f"{i} = {feature}"
    )

# ============================================================
# FINAL PASS / WARNING
# ============================================================

if agreement >= 0.99:

    print(
        "\nPASS: INT8 predictions agree "
        "with the Keras model."
    )

elif int8_accuracy >= (
    keras_accuracy - 0.02
):

    print(
        "\nPASS WITH SMALL DIFFERENCE:"
    )

    print(
        "INT8 accuracy is within 2 percentage "
        "points of the Keras model."
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "INT8 accuracy dropped significantly."
    )

print(
    "\n==============================================\n"
)