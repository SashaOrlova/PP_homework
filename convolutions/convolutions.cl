__kernel void matrix_conv(__global double * a, __global double * b, __global double * c, int n, int m)
{
   int i = get_global_id(0);
   int j = get_global_id(1);

   if (i >= n || j >= n)
      return;

   double sum = 0;
   int MN = (m - 1)/2;

   for (int k = -MN; k <= MN; ++k) {
      for (int l = -MN; l <= MN; ++l) {
        if (i + k >= n || j + l >= n || i + k < 0 || j + l < 0)
            continue;
        sum += a[(i + k) * n + j + l] * b[(k + MN) * m + l + MN];
       }
   }
   c[i * n + j] += sum;
}