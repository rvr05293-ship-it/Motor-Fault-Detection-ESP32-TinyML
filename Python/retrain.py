import numpy as np
import scipy.io as sio
import tensorflow as tf

normal = sio.loadmat('Normal_3.mat')['X100_DE_time'].flatten().astype(np.float32)

# Use 256 samples instead of 1024
INPUT_SIZE = 256

chunks = []
for i in range(0, len(normal) - INPUT_SIZE, 128):
    chunk = normal[i:i+INPUT_SIZE]
    chunk = chunk / (np.max(np.abs(chunk)) + 1e-8)
    chunks.append(chunk)

X_train = np.array(chunks)
print(f"Training samples: {X_train.shape}")

inp = tf.keras.Input(shape=(INPUT_SIZE,))
x   = tf.keras.layers.Dense(32, activation='relu')(inp)
x   = tf.keras.layers.Dense(8,  activation='relu')(x)
x   = tf.keras.layers.Dense(32, activation='relu')(x)
out = tf.keras.layers.Dense(INPUT_SIZE, activation='linear')(x)

model = tf.keras.Model(inp, out)
model.compile(optimizer='adam', loss='mse')
model.summary()

model.fit(X_train, X_train,
          epochs=100,
          batch_size=32,
          validation_split=0.1,
          shuffle=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
print(f"\nModel size: {len(tflite_model)/1024:.1f} KB")

with open('model.h', 'w') as f:
    f.write('#pragma once\n\n')
    f.write('#include <stdint.h>\n\n')
    f.write(f'const unsigned int model_tflite_len = {len(tflite_model)};\n')
    f.write('alignas(8) const unsigned char model_tflite[] = {\n  ')
    f.write(',\n  '.join(
        ', '.join(f'0x{b:02x}' for b in tflite_model[i:i+12])
        for i in range(0, len(tflite_model), 12)
    ))
    f.write('\n};\n')

print("✅ model.h generated!")

b007  = sio.loadmat('B007.mat')['X118_DE_time'].flatten().astype(np.float32)
ir007 = sio.loadmat('IR007.mat')['X105_DE_time'].flatten().astype(np.float32)
or007 = sio.loadmat('OR007.mat')['X130_DE_time'].flatten().astype(np.float32)

def get_mse(data, model, n=30):
    mses = []
    for _ in range(n):
        idx = np.random.randint(0, len(data)-INPUT_SIZE)
        chunk = data[idx:idx+INPUT_SIZE]
        chunk = chunk / (np.max(np.abs(chunk)) + 1e-8)
        pred = model.predict(chunk.reshape(1,-1), verbose=0)
        mse = np.mean((chunk - pred[0])**2)
        mses.append(mse)
    return np.mean(mses)

print("\n=== MSE Results ===")
n_mse  = get_mse(normal, model)
b_mse  = get_mse(b007,   model)
ir_mse = get_mse(ir007,  model)
or_mse = get_mse(or007,  model)

print(f"Normal MSE : {n_mse:.6f}")
print(f"Ball   MSE : {b_mse:.6f}")
print(f"Inner  MSE : {ir_mse:.6f}")
print(f"Outer  MSE : {or_mse:.6f}")
threshold = n_mse * 2
print(f"\n✅ Suggested threshold: {threshold:.6f}")
print(f"✅ Use INPUT_SIZE = {INPUT_SIZE} in Arduino")