#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

// static void matmul(float const *__restrict__ a, float const *__restrict__ b, float *__restrict__ res);
// void loopy_kernel(float const *__restrict__ b, float *__restrict__ a, float *__restrict__ res, uint8_t const *__restrict__ o)
// {
//   float a2[3 * 3];
//   float mat[3 * 3] = { 2.0f, 3.0f, 4.0f, 3.0f, 5.0f, 6.0f, 6.0f, 7.0f, 8.0f };
// //   float mat1[3 * 3] = { 4.0f, 6.0f, 8.0f, 6.0f, 10.0f, 12.0f, 12.0f, 24.0f, 16.0f };
//   if (*o == 0){
//     a2 = mat;
//   }
//   if (*o == 1){
//     a2 = mat1;
//   }
//   matmul(&(a2[0]), &(b[0]), &(res[0]));
// }

// static void matmul(float const *__restrict__ a, float const *__restrict__ b, float *__restrict__ res)
// {
//   for (int32_t i = 0; i <= 2; ++i)
//     for (int32_t j = 0; j <= 2; ++j)
//       res[j] = res[j] + a[3 * i + j] * b[i];
// }
// #include <stdint.h>
// #include <stdbool.h>

static void matmul(float const *__restrict__ a, float const *__restrict__ b, float *__restrict__ res);
void loopy_kernel(float const *__restrict__ b,  float *__restrict__ a2, float *__restrict__ res, int8_t const o)
{
//   float mat[3 * 3] = { 2.0f, 3.0f, 4.0f, 3.0f, 5.0f, 6.0f, 6.0f, 7.0f, 8.0f };
  float mat1[3 * 3] = { 4.0f, 6.0f, 8.0f, 6.0f, 10.0f, 12.0f, 12.0f, 24.0f, 16.0f };

  float mat0[3 * 3] = { 2.0f, 3.0f, 4.0f, 3.0f, 5.0f, 6.0f, 6.0f, 7.0f, 8.0f };
//   float mat1[3 * 3] = { 5.0f, 3.0f, 6.0f, 8.0f, 9.0f, 6.0f, 8.0f, 7.0f, 8.0f };

  switch (o) {
  case 0:
   a2 = mat0;break;
  case 1:
   a2 = mat1;break;
  default:
  break;
  }
  matmul(&(a2[0]), &(b[0]), &(res[0]));
}

static void matmul(float const *__restrict__ a, float const *__restrict__ b, float *__restrict__ res)
{
  for (int32_t i = 0; i <= 2; ++i)
    for (int32_t j = 0; j <= 2; ++j)
      res[j] = res[j] + a[3 * i + j] * b[i];
}

int main(void){
    int n = 3;
    float *Aptr = calloc(n*n, sizeof(float));
    Aptr[0] = 0;Aptr[1] = 1; Aptr[2]=0;
    Aptr[3] = 1;Aptr[4] = 0; Aptr[5]=0;
    Aptr[6] = 0;Aptr[7] = 0; Aptr[8]=1;

    float *Xptr = calloc(n, sizeof(float));
    Xptr[0] = 3;Xptr[1] = 4; Xptr[2] = 5;

    float *Yptr = calloc(n, sizeof(float));
    Yptr[0] = 0;Yptr[1] = 0; Yptr[2] = 0;
    uint8_t o = 1;
    float a2 = 0;
	loopy_kernel(Xptr, &a2, Yptr, o);
    printf("%f\n", Yptr[0]);
    printf("%f\n", Yptr[1]);
    printf("%f\n", Yptr[2]);


    free(Aptr);
    free(Xptr);
    free(Yptr);
	return 0;
}
