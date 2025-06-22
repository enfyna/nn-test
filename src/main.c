#include <math.h>
#include <stdio.h>

typedef struct {
    double ws[2];
    double bias;
} Perceptron;

double is[][2] = {
    { 0, 0 },
    { 1, 0 },
    { 0, 1 },
    { 1, 1 },
};

// double os[] = { 0, 0, 0, 1 }; // and
double os[] = { 0, 1, 1, 1 }; // or

double relu(double val)
{
    return val > 0 ? val : 0;
}

double sigmoid(double val)
{
    return val / (1 + fabsl(val));
}

double perceptron_predict(Perceptron* p, double* inputs)
{
    double result = p->bias;

    for (int i = 0; i < 2; i++) {
        result += inputs[i] * p->ws[i];
    }

    return relu(result);
}

void perceptron_fit(Perceptron* p)
{

    for (int repeat = 0; repeat < 1000; repeat++) {
        const double learning_rate = 0.01;

        double error = 0;

        for (int i = 0; i < 4; i++) {
            double target = os[i];
            double* inputs = is[i];
            double prediction = perceptron_predict(p, inputs);

            error = target - prediction;

            p->bias += learning_rate * error * 1;
            p->ws[0] += learning_rate * error * inputs[0];
            p->ws[1] += learning_rate * error * inputs[1];

            printf("%d: "
                   "err: %.2lf, ws[0]: %.2lf, ws[1]: %.2lf, b: %.2lf\n",
                repeat, error, p->ws[0], p->ws[1], p->bias);
        }
    }
}

int main(void)
{
    Perceptron p = { 0 };

    perceptron_fit(&p);

    printf("----------\n");
    printf("os[0]: %lf\n", perceptron_predict(&p, is[0]));
    printf("os[1]: %lf\n", perceptron_predict(&p, is[1]));
    printf("os[2]: %lf\n", perceptron_predict(&p, is[2]));
    printf("os[3]: %lf\n", perceptron_predict(&p, is[3]));

    return 0;
}
