#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"


#include "IndirectGridSample1D.h"
#include <torch/autograd.h>

#include <torch/csrc/autograd/custom_function.h>

using namespace Saiga;

#undef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#undef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit - 1), MAX(in, 0))
#define WITHIN_BOUNDS(x, W) (x >= 0 && x < W  )

static __global__ void IndirectGridSample1DForwardCUDAKernel(StaticDeviceTensor<float, 4> multi_grid,
                                                             StaticDeviceTensor<long, 1> index,
                                                             StaticDeviceTensor<float,2> uv,
                                                             StaticDeviceTensor<float,2> output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= output.sizes[0]) return;

    int C = multi_grid.sizes[1];
    int IW = multi_grid.sizes[2];

    int n = index(i);

    float ix = uv(i, 0);
    // normalize ix, iy from [-1, 1] to [0, IW-1] & [0, IH-1] 

    ix = ((ix + 1) /2 ) * (IW -1);

    int ix_tnw = floor((ix));
    int ix_tne = ix_tnw + 1;

    float tnw = (ix_tne - ix);
    float tne = (ix - ix_tnw);

    CLIP_COORDINATES(ix_tne, ix_tne, IW);
    CLIP_COORDINATES(ix_tnw, ix_tnw, IW);
    // printf("output i %d tnw %f tne %f  ix_tse %d ix_tnw %d ix %f \n", i, tnw, tne, ix_tne, ix_tnw, ix );

    for (int c = 0; c < C; ++c)
    {
        float out_val = 0;
        out_val       = 0;
        // printf("output i %d c %d ", i, c);
        if(WITHIN_BOUNDS(ix_tnw, IW))
        {
            // printf("gird %f tnw %f",multi_grid(n, c, ix_tnw, 0) , tnw);
            out_val += multi_grid(n, c, ix_tnw, 0) * tnw;
        }
        if(WITHIN_BOUNDS(ix_tne, IW))
        {
            // printf("gird %f tnw %f",multi_grid(n, c, ix_tne, 0) , tne);
            out_val += multi_grid(n, c, ix_tne, 0) * tne;
        }
        output(i, c) = out_val;
        // printf("\n");
        // printf("out put i %d c % d value %f tnw %f tne %f \n", i, c ,out_val, tnw, tne);
    }    
    
}

void IndirectGridSample1DForwardCUDA(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv,
                                     torch::Tensor& output)
{
    int num_samples = index.size(0);
    if(num_samples > 0)
    {
        IndirectGridSample1DForwardCUDAKernel<<<iDivUp(num_samples, 128), 128>>>(multi_grid, index, uv, output);
    }
}

static __global__ void IndirectGridSample1DBackwardCUDAKernel(StaticDeviceTensor<float, 4> multi_grid,
                                                              StaticDeviceTensor<long, 1> index,
                                                              StaticDeviceTensor<float, 2> uv,
                                                              StaticDeviceTensor<float, 4> grad_multi_grid,
                                                              StaticDeviceTensor<float, 2> grad_uv,
                                                              StaticDeviceTensor<float, 2> grad_output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= index.sizes[0]) return;
    int C  = multi_grid.sizes[1];
    int IW = multi_grid.sizes[2];

    // printf("IW is %d\n", IW);
    int n = index(i);

    float ix = uv(i, 0);
    ix = ((ix + 1) / 2) * (IW - 1);

    int ix_tnw = floor((ix));
    int ix_tne = ix_tnw + 1;

    float tnw = ix_tne - ix;
    float tne = ix - ix_tnw;

    int ix_tnw_cl, ix_tne_cl;
    CLIP_COORDINATES(ix_tnw, ix_tnw_cl, IW);
    CLIP_COORDINATES(ix_tne, ix_tne_cl, IW);

    float gix = 0;

    // printf("i %d tnw %f tne %f\n", i , tnw, tne);

    for (int c = 0; c < C; ++c)
    {
        float gradout = grad_output(i, c);
        atomicAdd(&grad_multi_grid(n, c,  ix_tnw_cl, 0), tnw * gradout);
        atomicAdd(&grad_multi_grid(n, c,  ix_tne_cl, 0), tne * gradout);

        float tnw_val = 0;
        if (WITHIN_BOUNDS(ix_tnw_cl, IW))
        {
            tnw_val = multi_grid(n, c, ix_tnw_cl, 0);
        }
        float tne_val = 0;
        if (WITHIN_BOUNDS(ix_tne_cl, IW))
        {
            // tne_val = input[n][c][iz_tne_cl][iy_tne_cl][ix_tne_cl];
            tne_val = multi_grid(n, c,  ix_tne_cl, 0);
        }
        // printf("i %d tnw_val %f tne_val %f\n",i,  tnw_val,tne_val);

        float m1 = -1;
        gix += m1 * tnw_val * (ix_tne - ix)  * gradout;
        gix += tne_val * (ix - ix_tnw) * gradout;

    }
    gix = gix * (IW - 1) / 2;
    // printf("i %d gix %f %f %f \n", i, gix , ix_tne - ix, ix - ix_tnw);
    grad_uv(i, 0) += gix;

}

void IndirectGridSample1DBackwardCUDA(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv,
                                      torch::Tensor& grad_multi_grid, torch::Tensor& grad_uv, torch::Tensor grad_output)
{
    int num_samples = index.size(0);
    if(num_samples > 0)
    {
        IndirectGridSample1DBackwardCUDAKernel<<<iDivUp(num_samples, 128), 128>>>(
            multi_grid, index, uv, grad_multi_grid, grad_uv, grad_output);
    }
}

namespace torch::autograd
{
    std::vector<torch::Tensor> IndirectGridSample1D::forward(AutogradContext* ctx, torch::Tensor multi_grid,
                                                            torch::Tensor index, torch::Tensor uv)
    {
        int num_samples      = index.size(0);
        int num_channels     = multi_grid.size(1);
        torch::Tensor output = torch::zeros({num_samples, num_channels}, multi_grid.options());
        if(multi_grid.is_cuda())
        {
            IndirectGridSample1DForwardCUDA(multi_grid, index, uv, output);
            CUDA_SYNC_CHECK_ERROR();
        } 
        else
        {
            CHECK(false);
        }
        std::vector<torch::Tensor> input;
        input.push_back(multi_grid);
        input.push_back(index);
        input.push_back(uv);
        ctx->save_for_backward(input);

        std::vector<torch::Tensor> result;
        result.push_back(output);
        return result;
    }

    std::vector<torch::Tensor> IndirectGridSample1D::backward(AutogradContext* ctx, std::vector<torch::Tensor> grad_output)
    {
        std::vector<torch::Tensor> input        = ctx->get_saved_variables();
        auto multi_grid                        = input[0];
        auto index                              = input[1];
        auto uv                                 = input[2];

        CHECK_EQ(grad_output.size(),1);

        auto grad_multi_grid = torch::zeros_like(multi_grid);
        auto grad_uv         = torch::zeros_like(uv);

        grad_multi_grid.set_requires_grad(1);
        grad_uv.set_requires_grad(1);

        if(multi_grid.is_cuda())
        {
            IndirectGridSample1DBackwardCUDA(multi_grid, index, uv, grad_multi_grid, grad_uv, grad_output[0]);
            CUDA_SYNC_CHECK_ERROR();
        }
        else
        {
            CHECK(false);
        }
        return {grad_multi_grid, torch::Tensor(), grad_uv};
    }
}

torch::Tensor IndirectGridSample1D(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv)
{
    CHECK(multi_grid.defined());
    CHECK(index.defined());
    CHECK(uv.defined());

    CHECK_EQ(multi_grid.dim(), 4);
    CHECK_EQ(index.dim(), 1);
    CHECK_EQ(uv.dim(), 2);
    auto result = torch::autograd::IndirectGridSample1D::apply(multi_grid, index, uv);
    CHECK_EQ(result.size(), 1);
    return result.front();
}