#include <EloquentTinyML.h>
#include "model.h"

#define INPUT_SIZE  256
#define OUTPUT_SIZE 256
#define ARENA_SIZE  (30 * 1024)

float input_data[INPUT_SIZE];
float output_data[OUTPUT_SIZE];

Eloquent::TinyML::TfLite<INPUT_SIZE, OUTPUT_SIZE, ARENA_SIZE> ml;

float threshold = 0.065000;

void setup() {
    Serial.begin(115200);
    pinMode(2, OUTPUT);
    digitalWrite(2, LOW);

    if (!ml.begin(model_tflite)) {
        Serial.println("ERROR");
        while (1);
    }
    Serial.println("READY");
}

void loop() {
    // Request data from Python
    Serial.println("SEND");

    int totalBytes = INPUT_SIZE * 4;
    int received = 0;
    uint8_t *ptr = (uint8_t*)input_data;

    unsigned long start = millis();

    // ✅ Receive data byte-by-byte
    while (received < totalBytes) {
        if (Serial.available()) {
            ptr[received++] = Serial.read();
        }

        if (millis() - start > 10000) {
            Serial.println("TIMEOUT");
            return;
        }
    }

    // Run inference
    ml.predict(input_data, output_data);

    float mse = 0.0;
    for (int i = 0; i < INPUT_SIZE; i++) {
        float diff = input_data[i] - output_data[i];

        if (!isnan(diff) && !isinf(diff)) {
            mse += diff * diff;
        }
    }

    mse /= INPUT_SIZE;

    if (isnan(mse) || isinf(mse)) mse = 0.0;

    if (mse > threshold) {
        digitalWrite(2, HIGH);
        Serial.print("FAULT:");
    } else {
        digitalWrite(2, LOW);
        Serial.print("NORMAL:");
    }

    Serial.println(mse, 6);

    delay(500);
}
