# Real-Time Motor Fault Detection Using Unsupervised Autoencoders on ESP32 Edge Devices

## Overview
This project implements a TinyML-based motor fault detection system using an unsupervised autoencoder deployed on an ESP32 microcontroller. The model is trained on healthy motor data and identifies abnormal motor conditions by analyzing input data and reconstruction error, enabling real-time fault detection on an edge device.

## Features
- Real-time fault detection on ESP32
- TinyML deployment using TensorFlow Lite
- Unsupervised Autoencoder model
- Edge AI inference
- Healthy vs Faulty motor condition classification
- Lightweight embedded implementation

## Tech Stack
- Python
- TensorFlow
- TensorFlow Lite
- C++
- Arduino IDE
- ESP32

## Repository Structure

Arduino/
- ESP32_Motor_AI.ino
- model.h

Python/
- train.py
- retrain.py
- train_esp32_compatible.py
- convert_quantized.py
- send_to_esp32.py

## Workflow
1. Prepare and preprocess motor condition data.
2. Train the autoencoder using healthy motor samples.
3. Convert the trained model to TensorFlow Lite.
4. Deploy the quantized model on the ESP32.
5. Classify incoming data as **Normal** or **Faulty** based on reconstruction error.

## Results
The deployed model successfully distinguishes healthy and faulty motor conditions on the ESP32 using TinyML-based edge inference.

## Author
**Raghul V R**  
B.Tech Electronics and Communication Engineering  
Amrita Vishwa Vidyapeetham
