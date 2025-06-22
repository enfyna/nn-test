#ifndef H_MAT
#define H_MAT

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define mat_num_t double

#define mat_at(mat, row, col) mat.es[(row) * (mat).stride + (col)]
#define mat_size(mat) (mat.rows * mat.cols)
#define mat_zero(mat) mat_fill(mat, 0)
#define mat_print(mat) mat_print_name(mat, #mat, 0)

#define nn_print(nn) nn_print_name(nn, #nn)

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    mat_num_t* es;
} Mat;

Mat mat_alloc(size_t rows, size_t cols);
Mat mat_row(Mat mat, size_t row);

void mat_sum(Mat dest, Mat add);
void mat_dot(Mat dest, Mat a, Mat b);

void mat_copy(Mat dest, Mat src);

void mat_sig(Mat mat);
void mat_randomize(Mat mat);
void mat_fill(Mat mat, mat_num_t val);
void mat_print_name(Mat mat, const char* name, size_t padding);

double sigmoid(double value);

typedef struct {
    mat_num_t rate;
    mat_num_t eps;
    size_t count;
    Mat* ws;
    Mat* bs;
    Mat* as; // the amount of activation is (count + 1)
} NN;

NN nn_alloc(size_t* arch, size_t arch_count);
mat_num_t nn_cost(NN nn, Mat ti, Mat to);
void nn_forward(NN nn);
void nn_print_name(NN nn, const char* name);
mat_num_t nn_cost(NN nn, Mat ti, Mat to);
void nn_finite_diff(NN nn, NN g, Mat ti, Mat to);
void nn_learn(NN nn, NN g);
void nn_randomize(NN nn, mat_num_t min, mat_num_t max);
int rand_range(int min, int max);

#endif // H_MAT

#ifdef MAT_IMPLEMENTATION

#include <math.h>

Mat mat_alloc(size_t rows, size_t cols)
{
    return (Mat) {
        .rows = rows,
        .cols = cols,
        .stride = cols,
        .es = (mat_num_t*)malloc(sizeof(mat_num_t) * rows * cols),
    };
}

void mat_copy(Mat dest, Mat src)
{
    assert(dest.cols == src.cols && dest.rows == src.rows && "Invalid Size");
    for (size_t i = 0; i < mat_size(dest); i++) {
        dest.es[i] = src.es[i];
    }
}

Mat mat_row(Mat mat, size_t row)
{
    return (Mat) {
        .rows = 1,
        .cols = mat.cols,
        .stride = mat.stride,
        .es = &mat_at(mat, row, 0),
    };
}

void mat_sum(Mat dest, Mat add)
{
    assert(dest.rows == add.rows && dest.cols == add.cols && "Invalid Size");
    for (size_t i = 0; i < mat_size(add); i++) {
        dest.es[i] += add.es[i];
    }
}

void mat_dot(Mat dest, Mat a, Mat b)
{
    assert(a.cols == b.rows && "Invalid input size");
    assert(dest.rows == a.rows && dest.cols == b.cols && "Invalid dest size");
    memset(dest.es, 0, mat_size(dest) * sizeof(mat_num_t));
    size_t n = a.cols;
    for (size_t i = 0; i < dest.rows; i++) {
        for (size_t j = 0; j < dest.cols; j++) {
            for (size_t k = 0; k < n; k++) {
                mat_at(dest, i, j) += mat_at(a, i, k) * mat_at(b, k, j);
                // printf("%f += %f*%f\n", mat_at(dest, i, j), mat_at(a, i, k), mat_at(b, k, j));
            }
            // printf("\n");
        }
    }
}

void mat_fill(Mat mat, mat_num_t val)
{
    for (size_t i = 0; i < mat_size(mat); i++) {
        mat.es[i] = val;
    }
}

void mat_randomize(Mat mat)
{
    for (size_t i = 0; i < mat_size(mat); i++) {
        mat.es[i] = (mat_num_t)rand() / RAND_MAX;
    }
}

double sigmoid(double value)
{
    return 1.f / (1.f + exp(-value));
}

void mat_sig(Mat mat)
{
    for (size_t i = 0; i < mat_size(mat); i++) {
        mat.es[i] = sigmoid(mat.es[i]);
    }
}

void mat_print_name(Mat mat, const char* name, size_t padding)
{
    printf("%*s%s: (r:%zu x c:%zu) [\n ", (int)padding, "", name, mat.rows, mat.cols);
    for (size_t i = 0; i < mat.rows; i++) {
        for (size_t j = 0; j < mat.cols; j++) {
            printf("%*s%6.3f, ", (int)padding, "", mat_at(mat, i, j));
        }
        printf("%*s\n ", (int)padding, "");
    }
    printf("\r%*s]\n", (int)padding, "");
}

// Neural Network

NN nn_alloc(size_t* arch, size_t arch_count)
{
    assert(arch_count > 0 && "Invalid arch_count\n");

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = calloc(sizeof *nn.ws, nn.count + 1);
    nn.bs = calloc(sizeof *nn.bs, nn.count + 1);
    nn.as = calloc(sizeof *nn.as, nn.count + 1);

    nn.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i <= nn.count; i++) {
        nn.ws[i] = mat_alloc(arch[i - 1], arch[i]);
        nn.as[i] = mat_alloc(nn.as[i - 1].rows, nn.ws[i].cols);
        nn.bs[i] = mat_alloc(nn.as[i].rows, nn.as[i].cols);
    }

    return nn;
}
void nn_print_name(NN nn, const char* name)
{
    printf("%s: [\n", name);

#define buf_size 32
    char buf[buf_size] = { 0 };
    snprintf(buf, buf_size, "%s.as[%d]", name, 0);
    mat_print_name(nn.as[0], buf, 4);

    for (size_t i = 1; i <= nn.count; i++) {
        snprintf(buf, buf_size, "%s.ws[%zu]", name, i);
        mat_print_name(nn.ws[i], buf, 4);
        snprintf(buf, buf_size, "%s.bs[%zu]", name, i);
        mat_print_name(nn.bs[i], buf, 4);
        snprintf(buf, buf_size, "%s.as[%zu]", name, i);
        mat_print_name(nn.as[i], buf, 4);
    }
    printf("]\n");
}

void nn_forward(NN nn)
{
    for (size_t i = 1; i <= nn.count; i++) {
        mat_dot(nn.as[i], nn.as[i - 1], nn.ws[i]);
        mat_sum(nn.as[i], nn.bs[i]);
        mat_sig(nn.as[i]);
    }
}

mat_num_t nn_cost(NN nn, Mat ti, Mat to)
{
    mat_num_t c = 0.0;
    assert(to.cols == nn.as[nn.count].cols);
    assert(ti.rows == to.rows);
    size_t n = ti.rows;

    for (size_t i = 0; i < n; i++) {
        Mat x = mat_row(ti, i);
        Mat y = mat_row(to, i);

        mat_copy(nn.as[0], x);
        nn_forward(nn);

        size_t q = to.cols;
        for (size_t j = 0; j < q; j++) {
            mat_num_t diff = mat_at(nn.as[nn.count], 0, j) - mat_at(y, 0, j);
            c += diff * diff;
            // printf("%f\n", diff);
        }
    }

    return c / n;
}

void nn_finite_diff(NN nn, NN g, Mat ti, Mat to)
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


void nn_learn(NN nn, NN g)
{
    for (size_t k = 1; k <= nn.count; k++) {
        for (size_t i = 0; i < nn.ws[k].rows; i++) {
            for (size_t j = 0; j < nn.ws[k].cols; j++) {
                mat_at(nn.ws[k], i, j) -= nn.rate * mat_at(g.ws[k], i, j);
            }
        }

        for (size_t i = 0; i < nn.bs[k].rows; i++) {
            for (size_t j = 0; j < nn.bs[k].cols; j++) {
                mat_at(nn.bs[k], i, j) -= nn.rate * mat_at(g.bs[k], i, j);
            }
        }
    }
}

int rand_range(int min, int max)
{
    return min + rand() / (RAND_MAX / (max - min + 1) + 1);
}

void nn_randomize(NN nn, mat_num_t min, mat_num_t max)
{
    for (size_t k = 1; k <= nn.count; k++) {
        for (size_t i = 0; i < nn.ws[k].rows; i++) {
            for (size_t j = 0; j < nn.ws[k].cols; j++) {
                mat_at(nn.ws[k], i, j) = rand_range(min, max);
            }
        }

        for (size_t i = 0; i < nn.bs[k].rows; i++) {
            for (size_t j = 0; j < nn.bs[k].cols; j++) {
                mat_at(nn.bs[k], i, j) = rand_range(min, max);
            }
        }
    }
}

#endif // MAT_IMPLEMENTATION
