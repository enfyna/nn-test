#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUTS 2
#define DATASET_SIZE 4
#define LEARNING_RATE 0.1
#define MOMENTUM 0.9
#define EPOCHS 100

float dataset[DATASET_SIZE][INPUTS] = {
    { 0, 0 },
    { 0, 1 },
    { 1, 0 },
    { 1, 1 }
};

int labels[DATASET_SIZE] = { 0, 1, 1, 1 }; // AND

float weights[INPUTS];
float bias;
float weight_velocity[INPUTS] = { 0 };
float bias_velocity = 0;

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float sigmoid_derivative(float y)
{
    return y * (1 - y);
}

void train(void)
{
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < DATASET_SIZE; i++) {
            // İleri yayılım
            float sum = bias;
            for (int j = 0; j < INPUTS; j++) {
                sum += dataset[i][j] * weights[j];
            }
            float predicted = sigmoid(sum);
            float error = labels[i] - predicted;

            // Momentumlu güncelleme
            float delta = error * sigmoid_derivative(predicted);

            for (int j = 0; j < INPUTS; j++) {
                weight_velocity[j] = MOMENTUM * weight_velocity[j] + LEARNING_RATE * delta * dataset[i][j];
                weights[j] += weight_velocity[j];
            }

            bias_velocity = MOMENTUM * bias_velocity + LEARNING_RATE * delta;
            bias += bias_velocity;
        }
    }
}

int predict(float x1, float x2)
{
    float sum = bias + x1 * weights[0] + x2 * weights[1];
    float output = sigmoid(sum);
    return output > 0.5 ? 1 : 0;
}

int main(void)
{
    // Rastgele başlat
    for (int i = 0; i < INPUTS; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    bias = ((float)rand() / RAND_MAX) * 2 - 1;

    printf("Eğitim oncesi tahminler:\n");
    for (int i = 0; i < DATASET_SIZE; i++) {
        printf("%d AND %d => %d\n",
            (int)dataset[i][0], (int)dataset[i][1],
            predict(dataset[i][0], dataset[i][1]));
    }

    train();

    printf("Eğitim sonrası tahminler:\n");
    for (int i = 0; i < DATASET_SIZE; i++) {
        printf("%d AND %d => %d\n",
            (int)dataset[i][0], (int)dataset[i][1],
            predict(dataset[i][0], dataset[i][1]));
    }

    return 0;
}
