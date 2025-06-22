#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAT_IMPLEMENTATION
#include "mat.h"

typedef struct {
    Mat a0; // input

    Mat w1;
    Mat b1;
    Mat a1; // intermediate activation

    Mat w2;
    Mat b2;
    Mat a2; // result
} Xor;

Xor xor_alloc(void)
{
    Xor x;

    x.a0 = mat_alloc(1, 2); // input
    mat_zero(x.a0);

    x.w1 = mat_alloc(2, 2);
    x.b1 = mat_alloc(1, 2);
    mat_zero(x.w1);
    mat_zero(x.b1);

    x.a1 = mat_alloc(1, 2); // intermediate activation
    mat_zero(x.a1);

    x.w2 = mat_alloc(2, 1);
    x.b2 = mat_alloc(1, 1);
    mat_zero(x.w2);
    mat_zero(x.b2);

    x.a2 = mat_alloc(1, 1); // result
    mat_zero(x.a2);

    return x;
}

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

void xor_forward(Xor m)
{
    mat_dot(m.a1, m.a0, m.w1);
    mat_sum(m.a1, m.b1);
    mat_sig(m.a1);

    mat_dot(m.a2, m.a1, m.w2);
    mat_sum(m.a2, m.b2);
    mat_sig(m.a2);
}

// ti => training input | to => training output
mat_num_t cost(Xor m, Mat ti, Mat to)
{
    mat_num_t c = 0.0;
    assert(to.cols == m.a2.cols);
    assert(ti.rows == to.rows);
    size_t n = ti.rows;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(m.a0, x);
        xor_forward(m);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++) {
            mat_num_t diff = mat_at(m.a2, 0, j) - mat_at(y, 0, j);
            c += diff * diff;
        }
    }

    return c / n;
}

void finite_diff(Xor m, Xor g, mat_num_t eps, Mat ti, Mat to)
{
    mat_num_t saved;

    mat_num_t c = cost(m, ti, to);

    for (size_t i = 0; i < m.w1.rows; i++) {
        for (size_t j = 0; j < m.w1.cols; j++) {
            saved = mat_at(m.w1, i, j);
            mat_at(m.w1, i, j) += eps;
            mat_at(g.w1, i, j) = (cost(m, ti, to) - c) / eps;
            mat_at(m.w1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b1.rows; i++) {
        for (size_t j = 0; j < m.b1.cols; j++) {
            saved = mat_at(m.b1, i, j);
            mat_at(m.b1, i, j) += eps;
            mat_at(g.b1, i, j) = (cost(m, ti, to) - c) / eps;
            mat_at(m.b1, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.w2.rows; i++) {
        for (size_t j = 0; j < m.w2.cols; j++) {
            saved = mat_at(m.w2, i, j);
            mat_at(m.w2, i, j) += eps;
            mat_at(g.w2, i, j) = (cost(m, ti, to) - c) / eps;
            mat_at(m.w2, i, j) = saved;
        }
    }

    for (size_t i = 0; i < m.b2.rows; i++) {
        for (size_t j = 0; j < m.b2.cols; j++) {
            saved = mat_at(m.b2, i, j);
            mat_at(m.b2, i, j) += eps;
            mat_at(g.b2, i, j) = (cost(m, ti, to) - c) / eps;
            mat_at(m.b2, i, j) = saved;
        }
    }
}

void xor_learn(Xor m, Xor g, mat_num_t rate)
{
    for (size_t i = 0; i < m.w1.rows; i++) {
        for (size_t j = 0; j < m.w1.cols; j++) {
            mat_at(m.w1, i, j) -= rate * mat_at(g.w1, i, j);
        }
    }

    for (size_t i = 0; i < m.b1.rows; i++) {
        for (size_t j = 0; j < m.b1.cols; j++) {
            mat_at(m.b1, i, j) -= rate * mat_at(g.b1, i, j);
        }
    }

    for (size_t i = 0; i < m.w2.rows; i++) {
        for (size_t j = 0; j < m.w2.cols; j++) {
            mat_at(m.w2, i, j) -= rate * mat_at(g.w2, i, j);
        }
    }

    for (size_t i = 0; i < m.b2.rows; i++) {
        for (size_t j = 0; j < m.b2.cols; j++) {
            mat_at(m.b2, i, j) -= rate * mat_at(g.b2, i, j);
        }
    }
}

int main(int argc, char** argv)
{
    (void)argc;
    (void)argv;

    srand(0);

    Xor m = xor_alloc(); // model
    Xor g = xor_alloc(); // gradient

    mat_randomize(m.w1);
    mat_randomize(m.w2);

    mat_randomize(m.b1);
    mat_randomize(m.b2);

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

    mat_print(ti);
    mat_print(to);

    mat_num_t eps = 1; // 1e-1;
    mat_num_t rate = 1; // 1e-1;

    for (int i = 0; i < 100 * 1000; i++) {
        finite_diff(m, g, eps, ti, to);
        xor_learn(m, g, rate);
        printf("%d: cost = %lf\n", i, cost(m, ti, to));
    }

    printf("------------\n");

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            mat_num_t inps[2] = { i, j };
            mat_copy(m.a0, (Mat) { .rows = 1, .cols = 2, .stride = 2, .es = inps });
            xor_forward(m);
            printf("%d ^ %d = %.3f\n", i, j, *m.a2.es);
        }
    }

    return 0;
}
