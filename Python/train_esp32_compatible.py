import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# =========================
# 1. PARAMETERS
# =========================
SEGMENT_SIZE = 1024
EPOCHS = 50
BATCH_SIZE = 32
MODEL_NAME = "esp32_vibration_final"

# =========================
# 2. LOAD & SEGMENT DATA
# =========================
def load_signal(filename):
    data = loadmat(filename)
    # Find the Vibration Data key (DE_time)
    key = next(k for k in data.keys() if "DE_time" in k)
    return data[key].flatten()

def segment_signal(signal):
    return np.array([signal[i:i+SEGMENT_SIZE] for i in range(0, len(signal) - SEGMENT_SIZE, SEGMENT_SIZE)])

print("Loading 3 healthy datasets...")
files = ["Normal_0.mat", "Normal_1.mat", "Normal_2.mat"] 
healthy_segments = []
for file in files:
    healthy_segments.extend(segment_signal(load_signal(file)))

healthy_segments = np.array(healthy_segments)

# Normalization
max_val = np.max(np.abs(healthy_segments))
healthy_segments = healthy_segments / max_val
X_train, X_test = train_test_split(healthy_segments, test_size=0.2)

# =========================
# 3. BUILD LIGHTWEIGHT AUTOENCODER
# =========================
input_layer = Input(shape=(SEGMENT_SIZE,))
# Reduced from 512 to 256 to fit in ESP32 Flash
x = Dense(256, activation="relu")(input_layer) 
x = Dense(64, activation="relu")(x)
bottleneck = Dense(32, activation="relu")(x) # Compressed representation
x = Dense(64, activation="relu")(bottleneck)
x = Dense(256, activation="relu")(x)
output_layer = Dense(SEGMENT_SIZE, activation="linear")(x)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

print("Training model...")
autoencoder.fit(X_train, X_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, X_test))

# =========================
# 4. STRICT INT8 CONVERSION
# =========================
print("\nPerforming Strict INT8 Quantization...")

def representative_data_gen():
    for i in range(100):
        # Calibration data for the 8-bit range
        yield [X_train[i].reshape(1, SEGMENT_SIZE).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(autoencoder)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen

# Force integer-only operations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Strip float metadata completely
converter.inference_input_type = tf.int8   
converter.inference_output_type = tf.int8  

tflite_model = converter.convert()

with open(f"{MODEL_NAME}.tflite", "wb") as f:
    f.write(tflite_model)

# =========================
# 5. GENERATE model_data.h
# =========================
print("Generating model_data.h...")
with open(f"{MODEL_NAME}.tflite", 'rb') as f:
    tflite_data = f.read()

with open("model_data.h", 'w') as f:
    f.write('#include <stdint.h>\n\n')
    f.write('const unsigned char esp32_vibration_ae_tflite[] __attribute__((aligned(16))) = {\n  ')
    for i, byte in enumerate(tflite_data):
        f.write(f'0x{byte:02x}, ')
        if (i + 1) % 12 == 0: f.write('\n  ')
    f.write('\n};\n\n')
    f.write(f'const int esp32_vibration_ae_tflite_len = {len(tflite_data)};\n')

print(f"\nSUCCESS!")
print(f"New Header Size: {len(tflite_data)/1024:.2f} KB")
print(f"IMPORTANT: Use Max Val {max_val} in your ESP32 code.")