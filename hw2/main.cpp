#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <random>
#include <iostream>

size_t const BLOCK_SIZE = 16;

void blocks_copy_arrays(double *from, int n, double *to, int m, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * n);
    cl::Buffer dev_b(context, CL_MEM_WRITE_ONLY, sizeof(double) * m);

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * n, from);

    int rounded_size = ((n + BLOCK_SIZE + 1) / BLOCK_SIZE) * BLOCK_SIZE;

    cl::Kernel kernel(program, "blocks_copy_arrays");
    cl::KernelFunctor matrix_conv(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    matrix_conv(dev_a, dev_b, n, m);

    queue.enqueueReadBuffer(dev_b, CL_TRUE, 0, sizeof(double) * m, to);
    to[0] = 0;
}

void summarize_arrays(double* from, int n, double* to, int m, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * n);
    cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * m);
    cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * m);

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * n, from);
    queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * m, to);

    int rounded_size = ((m + BLOCK_SIZE + 1) / BLOCK_SIZE) * BLOCK_SIZE;

    cl::Kernel kernel(program, "summarize_arrays");
    cl::KernelFunctor matrix_conv(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    matrix_conv(dev_a, dev_b, dev_c, m);

    queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * m, to);
}

void calc_prefix_sum(double *arr, int n, cl::Context &context, cl::Program &program, cl::CommandQueue &queue) {
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * n);
    cl::Buffer dev_b(context, CL_MEM_WRITE_ONLY, sizeof(double) * n);

    queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * n, arr);

    int rounded_size = ((n + BLOCK_SIZE + 1) / BLOCK_SIZE) * BLOCK_SIZE;

    cl::Kernel kernel(program, "calc_prefix_sum");
    cl::KernelFunctor matrix_conv(kernel, queue, cl::NullRange, cl::NDRange(rounded_size), cl::NDRange(BLOCK_SIZE));
    matrix_conv(dev_a, dev_b, cl::__local(sizeof(double) * BLOCK_SIZE), cl::__local(sizeof(double) * BLOCK_SIZE), n);

    queue.enqueueReadBuffer(dev_b, CL_TRUE, 0, sizeof(double) * n, arr);

    if (n > BLOCK_SIZE) {
        int m = (n + BLOCK_SIZE + 1) / BLOCK_SIZE;
        auto* sums = new double[m];
        blocks_copy_arrays(arr, n, sums, m, context, program, queue);
        calc_prefix_sum(sums, m, context, program, queue);
        summarize_arrays(sums, m, arr, n, context, program, queue);
    }
}

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    int n;
    std::cin >> n;

    auto* a = new double[n];
    for (int i = 0; i < n; i++)
        std::cin >> a[i];

    // create platform
    cl::Platform::get(&platforms);
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);


    // create context
    cl::Context context(devices);

    // create command queue
    cl::CommandQueue queue(context, devices[0]);

    // load opencl source
    std::ifstream cl_file("part_sum.cl");
    std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                  cl_string.length() + 1));
    // create program
    cl::Program program(context, source);

    program.build(devices);

    calc_prefix_sum(a, n, context, program, queue);

    std::cout << std::fixed << std::setprecision(3);

    for (int i = 0; i < n; i++)
        std::cout << a[i] << " ";

    return 0;
}