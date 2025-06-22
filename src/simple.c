#include <stdio.h>

double or_train[][2] = {
    { 0, 0 },
    { 1, 2 },
    { 2, 4 },
    { 3, 6 },
    { 4, 8 },
};
// const double train_count = (sizeof train / sizeof train[0]);
const double train_count = 5.0;

double cost(double w)
{
    double res = 0;
    for (size_t i = 0; i < train_count; i++) {
        double input = or_train[i][0];
        double actual = or_train[i][1];

        double prediction = w * input;
        double err = prediction - actual;
        res += err * err;
    }
    res /= train_count;
    return res;
}

double dcost(double w)
{
    double res = 0;
    for (size_t i = 0; i < train_count; i++) {
        double input = or_train[i][0];
        double actual = or_train[i][1];

        // sum((x*w - y)2) => 2(wx - y)x

        double prediction = w * input - actual;
        double err = prediction * 2 * input;
        res += err;
    }
    res /= train_count;
    return res;
}

int main(void)
{
    // y = w * i
    double wd = 100;
    double wf = 100;

    double h = 1e-3;
    double rate = 1e-1;

    for (int i = 0; i < 20; i++) {
        double learn_finite = (cost(wf + h) - cost(wf)) / h;
        double learn = dcost(wd);
        wf -= learn_finite * rate;
        wd -= learn * rate;
        // printf("w = %.2lf, learn = %.2lf, cost = %.2lf\n", wd, learn, cost(wd));
        printf("wf = %lf, wd = %lf\n", wf, wd);
    }

    return 0;
}
