#include <math.h>
#include <stdio.h>

// OR
double or_train[][3] = {
    { 0, 0, 0 },
    { 1, 0, 1 },
    { 0, 1, 1 },
    { 1, 1, 1 },
};

// AND
double and_train[][3] = {
    { 0, 0, 0 },
    { 1, 0, 0 },
    { 0, 1, 0 },
    { 1, 1, 1 },
};

const double train_count = ((float)sizeof or_train / sizeof or_train[0]);

double sigmoid(double value)
{
    return 1 / (1 + exp(-value));
}

double cost(double w1, double w2, double bs)
{
    double res = 0;
    for (size_t i = 0; i < train_count; i++) {
        double actual = and_train[i][2];
        double prediction = sigmoid(w1 * and_train[i][0] + w2 * and_train[i][1] + bs);

        double err = prediction - actual;
        res += err * err;
    }
    res /= train_count;
    return res;
}

int main(void)
{
    // y = w1 * i1 + w2 * i2 + bs
    double w1 = 2.0;
    double w2 = 2.0;
    double bs = 2.0;

    double h = 1e-3;
    double rate = 1e-1;

    for (int i = 0; i < 100000; i++) {
        double c = cost(w1, w2, bs);
        double learn_w1 = (cost(w1 + h, w2, bs) - c) / h;
        double learn_w2 = (cost(w1, w2 + h, bs) - c) / h;
        double learn_bs = (cost(w1, w2, bs + h) - c) / h;
        w1 -= learn_w1 * rate;
        w2 -= learn_w2 * rate;
        bs -= learn_bs * rate;
        printf("w1 = %.4lf, w2 = %.4lf, b = %.4lf, cost = %.8lf\n", w1, w2, bs, cost(w1, w2, bs));
    }

    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            printf("%d | %d = %lf\n", i, j, sigmoid(i * w1 + j * w2 + bs));
        }
    }

    return 0;
}
