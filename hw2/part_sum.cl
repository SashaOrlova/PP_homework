#define SWAP(a,b) {__local int * tmp=a; a=b; b=tmp;}

void make_sum(__local double *prev_sum, __local double *nxt_sum, uint local_id, uint size) {
    if (local_id >= size) {
        nxt_sum[local_id] = prev_sum[local_id] + prev_sum[local_id - size];
    } else {
        nxt_sum[local_id] = prev_sum[local_id];
    }
}

__kernel void calc_prefix_sum(__global double *input, __global double *output, __local double *prev_sum, __local double *nxt_sum, int size) {
    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint block_size = get_local_size(0);

    if (global_id < size)
        prev_sum[local_id] = nxt_sum[local_id] = input[global_id];
    barrier(CLK_LOCAL_MEM_FENCE);

    for(uint size = 1; size < block_size; size *= 2) {
        make_sum(prev_sum, nxt_sum, local_id, size);
        barrier(CLK_LOCAL_MEM_FENCE);
        SWAP(prev_sum, nxt_sum);
    }

    if (global_id < size)
        output[global_id] = prev_sum[local_id];
}

__kernel void blocks_copy_arrays(__global double *input, __global double *output, int input_size, int output_size) {
    uint global_id = get_global_id(0);
    uint block_size = get_local_size(0);
    uint ind  = global_id / block_size + 1;

    if (global_id < input_size && ind < output_size && 1 + global_id == ind * block_size)
        output[global_id / block_size + 1] = input[global_id];
}

__kernel void summarize_arrays(__global double *partial_input, __global double *input, __global double *output, int size) {
    uint global_id = get_global_id(0);
    uint block_size = get_local_size(0);
    uint part_index = global_id / block_size;

    if (global_id < size)
        output[global_id] = input[global_id] + partial_input[part_index];
}