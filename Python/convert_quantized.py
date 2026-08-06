import os

tflite_path = "esp32_vibration_final.tflite"
output_path = "model_data.h"

with open(tflite_path, 'rb') as f:
    data = f.read()

with open(output_path, 'w') as f:
    f.write('#include <stdint.h>\n\n')
    f.write('const unsigned char esp32_vibration_ae_tflite[] __attribute__((aligned(16))) = {\n  ')
    for i, byte in enumerate(data):
        f.write(f'0x{byte:02x}, ')
        if (i + 1) % 12 == 0: f.write('\n  ')
    f.write('\n};\n\n')
    f.write(f'const int esp32_vibration_ae_tflite_len = {len(data)};\n')

print(f"Header updated. New size: {len(data)/1024:.2f} KB")