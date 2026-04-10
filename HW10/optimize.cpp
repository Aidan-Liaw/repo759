//
// Created by aidan on 10/04/2026.
//

#include "optimize.h"

// ACKNOWLEDGEMENT: 95% of the work is my own, and 5% of the work is ChatGPT's
// ChatGPT was provided the code and was asked "Is the code correct?".
// It picked up that the exit condition for a for loop in optimize5 was wrong

data_t *get_vec_start(vec *v) {
     return v->data && v->len ? v->data : nullptr;
}

void optimize1(vec *v, data_t *dest) {
     size_t length = v->len;
     data_t *d = get_vec_start(v);
     data_t temp = IDENT;
     for (size_t i = 0; i < length; i++) {
          temp = temp OP d[i];
     }
     *dest = temp;
}

void optimize2(vec *v, data_t *dest) {
     size_t length = v->len;
     size_t limit = length - 1;
     data_t *d = get_vec_start(v);
     data_t x = IDENT;
     size_t i;
     // reduce 2 elements at a time
     for (i = 0; i < limit; i += 2) {
          x = (x OP d[i]) OP d[i + 1];
     }
     // Finish any remaining elements
     for (; i < length; i++) {
          x = x OP d[i];
     }
     *dest = x;
}
void optimize3(vec *v, data_t *dest) {
     size_t length = v->len;
     size_t limit = length - 1;
     data_t *d = get_vec_start(v);
     data_t x = IDENT;
     size_t i;
     /* reduce 2 elements at a time */
     for (i = 0; i < limit; i += 2) {
          x = x OP(d[i] OP d[i + 1]);
     }
     /* Finish any remaining elements */
     for (; i < length; i++) {
          x = x OP d[i];
     }
     *dest = x;
}
void optimize4(vec *v, data_t *dest) {
     size_t length = v->len;
     size_t limit = length - 1;
     data_t *d = get_vec_start(v);
     data_t x0 = IDENT;
     data_t x1 = IDENT;
     size_t i;
     // reduce 2 elements at a time
     for (i = 0; i < limit; i += 2) {
          x0 = x0 OP d[i];
          x1 = x1 OP d[i + 1];
     }
     // Finish any remaining elements
     for (; i < length; i++) {
          x0 = x0 OP d[i];
     }
     *dest = x0 OP x1;
}

// Unroll to degree L = 3, accumulate K = 3 parallel results
void optimize5(vec *v, data_t *dest) {
     size_t length = v->len;
     size_t limit = length;
     data_t *d = get_vec_start(v);
     data_t x0 = IDENT;
     data_t x1 = IDENT;
     data_t x2 = IDENT;
     size_t i;
     // reduce 3 elements at a time
     for (i = 0; i + 2 < limit; i += 3) {
          x0 = x0 OP d[i];
          x1 = x1 OP d[i + 1];
          x2 = x2 OP d[i + 2];
     }
     // Finish any remaining elements
     for (; i < length; i++) {
          x0 = x0 OP d[i];
     }
     *dest = x0 OP x1 OP x2;
}
