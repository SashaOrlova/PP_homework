#define __CL_ENABLE_EXCEPTIONS
#include <OpenCL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <iterator>

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {

        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0]);

        std::ifstream cl_file("convolutions.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(),
                                                      cl_string.length() + 1));

        cl::Program program(context, source);

        size_t const block_size = 16;
        program.build(devices,  "-D BLOCK_SIZE=16");

        size_t N = 0;
        size_t M = 0;
        std::cin >> N >> M;

        double a[N*N];
        double b[M*M];
        double c[N*N];

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;
                std::cin >> a[idx];
                c[idx] = 0;
            }
        }
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; ++j) {
                size_t idx = i * N + j;
                std::cin >> b[idx];
            }
        }


        cl::Buffer dev_a(context, CL_MEM_READ_ONLY, sizeof(double) * N * N);
        cl::Buffer dev_b(context, CL_MEM_READ_ONLY, sizeof(double) * M * M);
        cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, sizeof(double) * N * N);

        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N * N, a);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * M * M, b);

        cl::Kernel kernel(program, "matrix_conv");
        size_t t = ((N + block_size - 1)/block_size) * block_size;
        cl::KernelFunctor matrix_conv(kernel, queue, cl::NullRange,
                cl::NDRange(t, t),
                cl::NDRange(block_size, block_size));
        matrix_conv(dev_a, dev_b, dev_c, (int)N, (int)M);

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * N * N, c);

        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                size_t idx = i * N + j;
                std::cout << c[idx] << ' ';
            }
            std::cout << '\n';
        }
    } catch (cl::Error const &e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }
}