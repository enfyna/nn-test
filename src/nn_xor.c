#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

void nn_derivative(NN nn, NN g, Mat ti, Mat to)
{
    mat_num_t saved;

    mat_num_t c = nn_cost(nn, ti, to);

    for (size_t k = 1; k <= nn.count; k++) {
        for (size_t i = 0; i < nn.ws[k].rows; i++) {
            for (size_t j = 0; j < nn.ws[k].cols; j++) {
                saved = mat_at(nn.ws[k], i, j);
                mat_at(nn.ws[k], i, j) += nn.eps;
                mat_at(g.ws[k], i, j) = (nn_cost(nn, ti, to) - c) / nn.eps;
                mat_at(nn.ws[k], i, j) = saved;
            }
        }

        for (size_t i = 0; i < nn.bs[k].rows; i++) {
            for (size_t j = 0; j < nn.bs[k].cols; j++) {
                saved = mat_at(nn.bs[k], i, j);
                mat_at(nn.bs[k], i, j) += nn.eps;
                mat_at(g.bs[k], i, j) = (nn_cost(nn, ti, to) - c) / nn.eps;
                mat_at(nn.bs[k], i, j) = saved;
            }
        }
    }
}
// 2, 2, 1
// a: [1, 2] => [1, 2] => [1, 1]
// w:           [2, 2] => [2, 1]
// b:           [1, 2] => [1, 1]
// 4, 3, 2
// a: [1, 4] => [1, 3] => [1, 2]
// w:           [4, 3] => [3, 2]
// b:           [1, 3] => [1, 2]

mat_num_t train_data[] = {
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    1,
    1,
    1,
    0,
};

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    srand(time(NULL));

    size_t arch[] = { 2, 2, 1 };
    size_t arch_count = sizeof arch / sizeof arch[0];
    NN nn = nn_alloc(arch, arch_count);
    nn_randomize(nn, -10, 10);
    nn.eps = 1;
    nn.rate = 1;

    NN g = nn_alloc(arch, arch_count);

    size_t stride = 3;
    size_t n = sizeof train_data / sizeof train_data[0] / stride;

    Mat ti = {
        .rows = n,
        .cols = 2,
        .stride = stride,
        .es = train_data,
    };

    Mat to = {
        .rows = n,
        .cols = 1,
        .stride = stride,
        .es = train_data + 2,
    };

    // mat_num_t y = nn_cost(nn, ti, to);
    // while (y > 0.2) {
    // nn_randomize(nn, -10, 10);
    // y = nn_cost(nn, ti, to);
    // printf("Reshuffling... => New cost: %f\n", y);
    // usleep(100);
    // }

    size_t train_count = 1 * 100000;
    size_t print_count = train_count / 100;
    printf("Training for %zu iterations...\n", train_count);
    for (size_t i = 1; i <= train_count; i++) {
        nn_finite_diff(nn, g, ti, to);
        nn_learn(nn, g);
        if (i % print_count == 0) {
            printf("%8zu: %f\n", i, nn_cost(nn, ti, to));
        }
        usleep(100);
    }

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(nn.as[0], x);
        nn_forward(nn);

        mat_print(x);
        printf("=> %f == %f\n", mat_at(nn.as[nn.count], 0, 0), mat_at(y, 0, 0));
        printf("=======================\n");
    }

    nn_print(nn);

    return 0;
}
