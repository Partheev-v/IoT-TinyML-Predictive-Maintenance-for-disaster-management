import os

TFLITE_FILE = "model/fan_fault_model_v2.tflite"
HEADER_FILE = "TinyML_Fan_ESP32/model_data.h"

print("=" * 60)
print(" TFLITE -> ESP32 C HEADER")
print("=" * 60)

if not os.path.exists(TFLITE_FILE):
    print("ERROR: TFLite model not found:")
    print(TFLITE_FILE)
    raise SystemExit(1)

with open(TFLITE_FILE, "rb") as f:
    model = f.read()

print("\nTFLite model loaded.")
print("Model size:", len(model), "bytes")

os.makedirs(
    os.path.dirname(HEADER_FILE),
    exist_ok=True
)

with open(HEADER_FILE, "w") as f:

    f.write("#ifndef MODEL_DATA_H\n")
    f.write("#define MODEL_DATA_H\n\n")

    f.write("// ============================================================\n")
    f.write("// TinyML Fan Fault Detection - Model V2\n")
    f.write("// Automatically generated from fan_fault_model_v2.tflite\n")
    f.write("// ============================================================\n\n")

    f.write("#include <stdint.h>\n\n")

    f.write("const unsigned char model_data[] = {\n")

    for i, byte in enumerate(model):

        if i % 12 == 0:
            f.write("  ")

        f.write(f"0x{byte:02x}")

        if i != len(model) - 1:
            f.write(", ")

        if i % 12 == 11:
            f.write("\n")

    if len(model) % 12 != 0:
        f.write("\n")

    f.write("};\n\n")

    f.write(
        "const unsigned int model_data_len = "
        f"{len(model)};\n\n"
    )

    f.write("#endif // MODEL_DATA_H\n")

print("\nHeader generated successfully:")
print(HEADER_FILE)

print("\nModel bytes:", len(model))

print("\n" + "=" * 60)
print(" COMPLETE")
print("=" * 60)
