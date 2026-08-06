import serial
import numpy as np
import scipy.io as sio
import time

PORT     = 'COM3'
BAUDRATE = 115200

normal = sio.loadmat('Normal_3.mat')['X100_DE_time'].flatten().astype(np.float32)
b007   = sio.loadmat('B007.mat')['X118_DE_time'].flatten().astype(np.float32)
ir007  = sio.loadmat('IR007.mat')['X105_DE_time'].flatten().astype(np.float32)
or007  = sio.loadmat('OR007.mat')['X130_DE_time'].flatten().astype(np.float32)

datasets = [
    ("NORMAL - Healthy", normal, "NORMAL"),
    ("FAULT  - Ball",    b007,   "FAULT"),
    ("FAULT  - Inner",   ir007,  "FAULT"),
    ("FAULT  - Outer",   or007,  "FAULT"),
]

def get_chunk(data):
    idx = np.random.randint(0, len(data) - 256)
    chunk = data[idx:idx+256]
    chunk = chunk / (np.max(np.abs(chunk)) + 1e-8)
    return chunk.astype(np.float32)

print("Connecting to ESP32...")
ser = serial.Serial(PORT, BAUDRATE, timeout=15)
time.sleep(2)
ser.flushInput()

print("Waiting for READY... press EN button on ESP32")
while True:
    line = ser.readline().decode(errors='ignore').strip()
    if line:
        print(f"ESP32: {line}")
    if "READY" in line:
        break

print("\nStarting test...\n")
print("=" * 50)

ds_index = 0
sample_count = [0, 0, 0, 0]
count = 0

print(f"\n--- Sending {datasets[0][0]} data ---")

while ds_index < len(datasets):
    label, data, expected = datasets[ds_index]

    line = ser.readline().decode(errors='ignore').strip()

    if "SEND" in line:
        count += 1
        chunk = get_chunk(data)
        # Send immediately with no delay
        bytes_sent = ser.write(chunk.tobytes())
        ser.flush()
        print(f"  Sent {bytes_sent} bytes for [{count}] {label}")

    elif "NORMAL" in line or "FAULT" in line:
        detected = "FAULT" if "FAULT" in line else "NORMAL"
        correct = "CORRECT" if detected == expected else "WRONG"
        print(f"  [{count}] {detected} | {line} | {correct} | Expected: {expected}")

        sample_count[ds_index] += 1
        if sample_count[ds_index] >= 5:
            ds_index += 1
            if ds_index < len(datasets):
                print(f"\n--- Sending {datasets[ds_index][0]} data ---")

    elif "TIMEOUT" in line:
        print(f"  Timeout! Retrying...")

    elif line:
        print(f"  ESP32: {line}")

print("\nTest complete!")
ser.close()
