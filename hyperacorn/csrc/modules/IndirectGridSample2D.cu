
#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"



#include "IndirectGridSample2D.h"
#include <torch/autograd.h>

#include <torch/csrc/autograd/custom_function.h>

using namespace Saiga;

#undef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#undef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define CLIP_COORDINATES(in, out, clip_limit) out = MIN((clip_limit - 1), MAX(in, 0))
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H )

static __global__ void IndirectGridSample2DForwardCUDAKernel(StaticDeviceTensor<float, 4> multi_grid,
                                                             StaticDeviceTensor<long, 1> index,
                                                             StaticDeviceTensor<float, 2> uv,
                                                             StaticDeviceTensor<float, 2> output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= output.sizes[0]) return;

    int C = multi_grid.sizes[1];
    int IH = multi_grid.sizes[2];
    int IW = multi_grid.sizes[3];

    // printf("IH %d IW %d\n", IH, IW);
    int n = index(i);
    // float* grid = multi_grid_ptr + index_ptr[i] * multi_grid.stride(0);

    float ix = uv(i,0);
    float iy = uv(i,1);

    // normalize ix, iy from [-1, 1] to [0, IW-1] & [0, IH-1]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);

    int ix_tnw = floor((ix));
    int iy_tnw = floor((iy));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;

    // int ix_bnw = ix_tnw;
    // int iy_bnw = iy_tnw;

    // int ix_bne = ix_tnw + 1;
    // int iy_bne = iy_tnw;

    // int ix_bsw = ix_tnw;
    // int iy_bsw = iy_tnw + 1;

    // int ix_bse = ix_tnw + 1;
    // int iy_bse = iy_tnw + 1;

    // get surfaces to each neighbor:
    // float tnw = (ix_bse - ix) * (iy_bse - iy) ;
    // float tne = (ix - ix_bsw) * (iy_bsw - iy) ;
    // float tsw = (ix_bne - ix) * (iy - iy_bne) ;
    // float tse = (ix - ix_bnw) * (iy - iy_bnw) ;
    // float bnw = (ix_tse - ix) * (iy_tse - iy) ;
    // float bne = (ix - ix_tsw) * (iy_tsw - iy) ;
    // float bsw = (ix_tne - ix) * (iy - iy_tne) ;
    // float bse = (ix - ix_tnw) * (iy - iy_tnw) ;

    float tnw = (ix_tse - ix) * (iy_tse - iy) ;
    float tne = (ix - ix_tsw) * (iy_tsw - iy) ;
    float tsw = (ix_tne - ix) * (iy - iy_tne) ;
    float tse = (ix - ix_tnw) * (iy - iy_tnw) ;
    CLIP_COORDINATES(ix_tnw, ix_tnw, IW);
    CLIP_COORDINATES(iy_tnw, iy_tnw, IH);
    CLIP_COORDINATES(ix_tne, ix_tne, IW);
    CLIP_COORDINATES(iy_tne, iy_tne, IH);
    CLIP_COORDINATES(ix_tsw, ix_tsw, IW);
    CLIP_COORDINATES(iy_tsw, iy_tsw, IH);
    CLIP_COORDINATES(ix_tse, ix_tse, IW);
    CLIP_COORDINATES(iy_tse, iy_tse, IH);
    // CLIP_COORDINATES(ix_bnw, ix_bnw, IW);
    // CLIP_COORDINATES(iy_bnw, iy_bnw, IH);
    // CLIP_COORDINATES(ix_bne, ix_bne, IW);
    // CLIP_COORDINATES(iy_bne, iy_bne, IH);
    // CLIP_COORDINATES(ix_bsw, ix_bsw, IW);
    // CLIP_COORDINATES(iy_bsw, iy_bsw, IH);
    // CLIP_COORDINATES(ix_bse, ix_bse, IW);
    // CLIP_COORDINATES(iy_bse, iy_bse, IH);

    // printf("output i %d tnw %f tne %f tsw %f tse %f ix_tse %d ix %f iy_tse %d iy %f\n", i, tnw, tne, tsw, tse, ix_tse, ix ,iy_tse, iy);
    for (int c = 0; c < C; ++c)
    {
        float out_val = 0;
        out_val       = 0;
        // printf("output i %d c %d ", i, c);

        if (WITHIN_BOUNDS(ix_tnw, iy_tnw,  IH, IW))
        {
            // printf("gird %f tnw %f",multi_grid(n, c, iy_tnw, ix_tnw) , tnw);

            // THTensor_fastGet5d(input, n, c, z, y, x);
            // out_val += input[n][c][iz_tnw][iy_tnw][ix_tnw] * tnw;
            out_val += multi_grid(n, c, iy_tnw, ix_tnw) * tnw;
        }
        if (WITHIN_BOUNDS(ix_tne, iy_tne,  IH, IW))
        {
            // printf("gird %f tne %f",multi_grid(n, c,  iy_tne, ix_tne) , tne);

            //                out_val += input[n][c][iz_tne][iy_tne][ix_tne] * tne;
            out_val += multi_grid(n, c,  iy_tne, ix_tne) * tne;
        }
        if (WITHIN_BOUNDS(ix_tsw, iy_tsw, IH, IW))
        {
            // printf("gird %f tsw %f",multi_grid(n, c, iy_tsw, ix_tsw) , tsw);

            //                out_val += input[n][c][iz_tsw][iy_tsw][ix_tsw] * tsw;
            out_val += multi_grid(n, c, iy_tsw, ix_tsw) * tsw;
        }
        if (WITHIN_BOUNDS(ix_tse, iy_tse, IH, IW))
        {
            // printf("gird %f tse %f",multi_grid(n, c, iy_tse, ix_tse) , tse);

            //                out_val += input[n][c][iz_tse][iy_tse][ix_tse] * tse;
            out_val += multi_grid(n, c, iy_tse, ix_tse) * tse;
        }
        // if (WITHIN_BOUNDS(ix_bnw, iy_bnw, IH, IW))
        // {
        //     //                out_val += input[n][c][iz_bnw][iy_bnw][ix_bnw] * bnw;
        //     out_val += multi_grid(n, c, iy_bnw, ix_bnw) * bnw;
        // }
        // if (WITHIN_BOUNDS(ix_bne, iy_bne, IH, IW))
        // {
        //     //                out_val += input[n][c][iz_bne][iy_bne][ix_bne] * bne;
        //     out_val += multi_grid(n, c,  iy_bne, ix_bne) * bne;
        // }
        // if (WITHIN_BOUNDS(ix_bsw, iy_bsw,  IH, IW))
        // {
        //     //                out_val += input[n][c][iz_bsw][iy_bsw][ix_bsw] * bsw;
        //     out_val += multi_grid(n, c,  iy_bsw, ix_bsw) * bsw;
        // }
        // if (WITHIN_BOUNDS(ix_bse, iy_bse, IH, IW))
        // {
        //     //                out_val += input[n][c][iz_bse][iy_bse][ix_bse] * bse;
        //     out_val += multi_grid(n, c,  iy_bse, ix_bse) * bse;
        // }
        output(i, c) = out_val;
        // printf("\n");
        // cout <<"out put i " << i << " c " << c << "value " << out_val;

    }
}

void IndirectGridSample2DForwardCUDA(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv,
                                     torch::Tensor& output)
{
    int num_samples = index.size(0);
    if (num_samples > 0)
    {
        IndirectGridSample2DForwardCUDAKernel<<<iDivUp(num_samples, 128), 128>>>(multi_grid, index, uv, output);
    }
}


static __global__ void IndirectGridSample2DBackwardCUDAKernel(StaticDeviceTensor<float, 4> multi_grid,
                                                              StaticDeviceTensor<long, 1> index,
                                                              StaticDeviceTensor<float, 2> uv,
                                                              StaticDeviceTensor<float, 4> grad_multi_grid,
                                                              StaticDeviceTensor<float, 2> grad_uv,
                                                              StaticDeviceTensor<float, 2> grad_output)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= index.sizes[0]) return;
    int C  = multi_grid.sizes[1];
    int IH = multi_grid.sizes[2];
    int IW = multi_grid.sizes[3];

    int n = index(i);
    // float* grid = multi_grid_ptr + index_ptr[i] * multi_grid.stride(0);

    float ix = uv(i, 0);
    float iy = uv(i, 1);
    // float iz = uv(i, 2);

    // normalize ix, iy, iz from [-1, 1] to [0, IW-1] & [0, IH-1] & [0, ID-1]
    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    // iz = ((iz + 1) / 2) * (ID - 1);

    int ix_tnw = floor((ix));
    int iy_tnw = floor((iy));
    // int iz_tnw = floor((iz));

    int ix_tne = ix_tnw + 1;
    int iy_tne = iy_tnw;
    // int iz_tne = iz_tnw;

    int ix_tsw = ix_tnw;
    int iy_tsw = iy_tnw + 1;
    // int iz_tsw = iz_tnw;

    int ix_tse = ix_tnw + 1;
    int iy_tse = iy_tnw + 1;
    // int iz_tse = iz_tnw;

    // int ix_bnw = ix_tnw;
    // int iy_bnw = iy_tnw;
    // int iz_bnw = iz_tnw + 1;

    // int ix_bne = ix_tnw + 1;
    // int iy_bne = iy_tnw;
    // int iz_bne = iz_tnw + 1;

    // int ix_bsw = ix_tnw;
    // int iy_bsw = iy_tnw + 1;
    // int iz_bsw = iz_tnw + 1;

    // int ix_bse = ix_tnw + 1;
    // int iy_bse = iy_tnw + 1;
    // int iz_bse = iz_tnw + 1;

    // get surfaces to each neighbor:
    // float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    // float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    // float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    // float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    // float bnw = (ix_tse - ix) * (iy_tse - iy) ;
    // float bne = (ix - ix_tsw) * (iy_tsw - iy) ;
    // float bsw = (ix_tne - ix) * (iy - iy_tne) ;
    // float bse = (ix - ix_tnw) * (iy - iy_tnw) ;

    float tnw = (ix_tse - ix) * (iy_tse - iy) ;
    float tne = (ix - ix_tsw) * (iy_tsw - iy) ;
    float tsw = (ix_tne - ix) * (iy - iy_tne) ;
    float tse = (ix - ix_tnw) * (iy - iy_tnw) ;

    int ix_tnw_cl, iy_tnw_cl, ix_tne_cl, iy_tne_cl;
    int ix_tsw_cl, iy_tsw_cl, ix_tse_cl, iy_tse_cl;
    // int ix_bnw_cl, iy_bnw_cl, ix_bne_cl, iy_bne_cl;
    // int ix_bsw_cl, iy_bsw_cl, ix_bse_cl, iy_bse_cl;

    // printf("i %d tnw %f tne %f tsw %f tse %f\n", i , tnw, tne, tsw, tse);


    CLIP_COORDINATES(ix_tnw, ix_tnw_cl, IW);
    CLIP_COORDINATES(iy_tnw, iy_tnw_cl, IH);
    // CLIP_COORDINATES(iz_tnw, iz_tnw_cl, ID);
    CLIP_COORDINATES(ix_tne, ix_tne_cl, IW);
    CLIP_COORDINATES(iy_tne, iy_tne_cl, IH);
    // CLIP_COORDINATES(iz_tne, iz_tne_cl, ID);
    CLIP_COORDINATES(ix_tsw, ix_tsw_cl, IW);
    CLIP_COORDINATES(iy_tsw, iy_tsw_cl, IH);
    // CLIP_COORDINATES(iz_tsw, iz_tsw_cl, ID);
    CLIP_COORDINATES(ix_tse, ix_tse_cl, IW);
    CLIP_COORDINATES(iy_tse, iy_tse_cl, IH);
    // CLIP_COORDINATES(iz_tse, iz_tse_cl, ID);
    // CLIP_COORDINATES(ix_bnw, ix_bnw_cl, IW);
    // CLIP_COORDINATES(iy_bnw, iy_bnw_cl, IH);
    // CLIP_COORDINATES(iz_bnw, iz_bnw_cl, ID);
    // CLIP_COORDINATES(ix_bne, ix_bne_cl, IW);
    // CLIP_COORDINATES(iy_bne, iy_bne_cl, IH);
    // CLIP_COORDINATES(iz_bne, iz_bne_cl, ID);
    // CLIP_COORDINATES(ix_bsw, ix_bsw_cl, IW);
    // CLIP_COORDINATES(iy_bsw, iy_bsw_cl, IH);
    // CLIP_COORDINATES(iz_bsw, iz_bsw_cl, ID);
    // CLIP_COORDINATES(ix_bse, ix_bse_cl, IW);
    // CLIP_COORDINATES(iy_bse, iy_bse_cl, IH);
    // CLIP_COORDINATES(iz_bse, iz_bse_cl, ID);

    // printf("locate info %d %d %d %d %d %d %d %d",ix_tnw_cl, iy_tnw_cl, ix_tne_cl, iy_tne_cl, ix_tsw_cl, iy_tsw_cl, ix_tse_cl, iy_tse_cl);
    float gix = 0;
    float giy = 0;
    // float giz = 0;

    for (int c = 0; c < C; ++c)
    {
        float gradout = grad_output(i, c);

        atomicAdd(&grad_multi_grid(n, c,  iy_tnw_cl, ix_tnw_cl), tnw * gradout);
        atomicAdd(&grad_multi_grid(n, c,  iy_tne_cl, ix_tne_cl), tne * gradout);
        atomicAdd(&grad_multi_grid(n, c,  iy_tsw_cl, ix_tsw_cl), tsw * gradout);
        atomicAdd(&grad_multi_grid(n, c,  iy_tse_cl, ix_tse_cl), tse * gradout);
        // atomicAdd(&grad_multi_grid(n, c,  iy_bnw_cl, ix_bnw_cl), bnw * gradout);
        // atomicAdd(&grad_multi_grid(n, c,  iy_bne_cl, ix_bne_cl), bne * gradout);
        // atomicAdd(&grad_multi_grid(n, c,  iy_bsw_cl, ix_bsw_cl), bsw * gradout);
        // atomicAdd(&grad_multi_grid(n, c,  iy_bse_cl, ix_bse_cl), bse * gradout);

        // output_ptr[i * C + c] = out_val;

        // calculate gradGrid
        float tnw_val = 0;
        if (WITHIN_BOUNDS(ix_tnw_cl, iy_tnw_cl,  IH, IW))
        {
            tnw_val = multi_grid(n, c, iy_tnw_cl, ix_tnw_cl);
        }
        float tne_val = 0;
        if (WITHIN_BOUNDS(ix_tne_cl, iy_tne_cl, IH, IW))
        {
            // tne_val = input[n][c][iz_tne_cl][iy_tne_cl][ix_tne_cl];
            tne_val = multi_grid(n, c,  iy_tne_cl, ix_tne_cl);
        }
        float tsw_val = 0;
        if (WITHIN_BOUNDS(ix_tsw_cl, iy_tsw_cl, IH, IW))
        {
            //                tsw_val = input[n][c][iz_tsw_cl][iy_tsw_cl][ix_tsw_cl];
            tsw_val = multi_grid(n, c,  iy_tsw_cl, ix_tsw_cl);
        }

        float tse_val = 0;
        if (WITHIN_BOUNDS(ix_tse_cl, iy_tse_cl, IH, IW))
        {
            //                tse_val = input[n][c][iz_tse_cl][iy_tse_cl][ix_tse_cl];
            tse_val = multi_grid(n, c,  iy_tse_cl, ix_tse_cl);
        }
        // printf("i %d tnw_val %f tne_val %f tsw_val %f tse_val %f\n", i, tnw_val, tne_val, tsw_val, tse_val);
        // float bnw_val = 0;
        // if (WITHIN_BOUNDS(ix_bnw_cl, iy_bnw_cl, iz_bnw_cl, ID, IH, IW))
        // {
        //     //                bnw_val = input[n][c][iz_bnw_cl][iy_bnw_cl][ix_bnw_cl];
        //     bnw_val = multi_grid(n, c, iz_bnw_cl, iy_bnw_cl, ix_bnw_cl);
        // }
        // float bne_val = 0;
        // if (WITHIN_BOUNDS(ix_bne_cl, iy_bne_cl, iz_bne_cl, ID, IH, IW))
        // {
        //     //                bne_val = input[n][c][iz_bne_cl][iy_bne_cl][ix_bne_cl];
        //     bne_val = multi_grid(n, c, iz_bne_cl, iy_bne_cl, ix_bne_cl);
        // }
        // float bsw_val = 0;
        // if (WITHIN_BOUNDS(ix_bsw_cl, iy_bsw_cl, iz_bsw_cl, ID, IH, IW))
        // {
        //     //                bsw_val = input[n][c][iz_bsw_cl][iy_bsw_cl][ix_bsw_cl];
        //     bsw_val = multi_grid(n, c, iz_bsw_cl, iy_bsw_cl, ix_bsw_cl);
        // }
        // float bse_val = 0;
        // if (WITHIN_BOUNDS(ix_bse_cl, iy_bse_cl, iz_bse_cl, ID, IH, IW))
        // {
        //     //                bse_val = input[n][c][iz_bse_cl][iy_bse_cl][ix_bse_cl];
        //     bse_val = multi_grid(n, c, iz_bse_cl, iy_bse_cl, ix_bse_cl);
        // }

        float m1 = -1;
        // gix += m1 * tnw_val * (iy_bse - iy) * gradout;
        // gix += tne_val * (iy_bsw - iy)  * gradout;
        // gix += m1 * tsw_val * (iy - iy_bne) * gradout;
        // gix += tse_val * (iy - iy_bnw)  * gradout;
        gix += m1 * tnw_val * (iy_tse - iy)  * gradout;
        gix += tne_val * (iy_tsw - iy)  * gradout;
        gix += m1 * tsw_val * (iy - iy_tne)  * gradout;
        gix += tse_val * (iy - iy_tnw)  * gradout;


        // giy += m1 * tnw_val * (ix_bse - ix)  * gradout;
        // giy += m1 * tne_val * (ix - ix_bsw)  * gradout;
        // giy += tsw_val * (ix_bne - ix)  * gradout;
        // giy += tse_val * (ix - ix_bnw)  * gradout;
        giy += m1 * tnw_val * (ix_tse - ix)  * gradout;
        giy += m1 * tne_val * (ix - ix_tsw)  * gradout;
        giy += tsw_val * (ix_tne - ix)  * gradout;
        giy += tse_val * (ix - ix_tnw)  * gradout;
        // printf("i add gix %f giy %f number %f %f %f %f\n", gix, giy, tnw_val * (iy_tse - iy), tne_val * (iy_tsw - iy), tsw_val * (iy - iy_tne) ,   tse_val * (iy - iy_tnw) );

        // giz += m1 * tnw_val * (ix_bse - ix) * (iy_bse - iy) * gradout;
        // giz += m1 * tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gradout;
        // giz += m1 * tsw_val * (ix_bne - ix) * (iy - iy_bne) * gradout;
        // giz += m1 * tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gradout;
        // giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gradout;
        // giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gradout;
        // giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gradout;
        // giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gradout;
    }

    // un-normalize gradGrid values back to [-1, 1] constraints
    gix = gix * (IW - 1) / 2;
    giy = giy * (IH - 1) / 2;
    // giz = giz * (ID - 1) / 2;
    // printf("i %d gix %f giy %f %f %f %f %f\n", i, gix, giy,iy_tse - iy, iy_tsw - iy,iy - iy_tne, iy - iy_tnw);
    // printf("i %d gix %f giy %f %d %f %f\n", i, gix, giy,ix_tse , ix, ix_tse - ix);

    // one thread per sample so don't need an atomic add
    grad_uv(i, 0) += gix;
    grad_uv(i, 1) += giy;
    // grad_uv(i, 2) += giz;
}

void IndirectGridSample2DBackwardCUDA(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv,
                                      torch::Tensor& grad_multi_grid, torch::Tensor& grad_uv, torch::Tensor grad_output)
{
    int num_samples = index.size(0);
    if (num_samples > 0)
    {
        IndirectGridSample2DBackwardCUDAKernel<<<iDivUp(num_samples, 128), 128>>>(
            multi_grid, index, uv, grad_multi_grid, grad_uv, grad_output);
    }
}

namespace torch::autograd
{
    std::vector<torch::Tensor> IndirectGridSample2D::forward(AutogradContext* ctx, torch::Tensor multi_grid,
                                                         torch::Tensor index, torch::Tensor uv)
    {
        int num_samples      = index.size(0);
        int num_channels     = multi_grid.size(1);
        torch::Tensor output = torch::zeros({num_samples, num_channels}, multi_grid.options());
        // if (multi_grid.is_cpu())
        // {
        //     IndirectGridSample3DForwardCPU(multi_grid, index, uv, output);
        // }
        // else
        if (multi_grid.is_cuda())
        {
            IndirectGridSample2DForwardCUDA(multi_grid, index, uv, output);
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

    std::vector<torch::Tensor> IndirectGridSample2D::backward(AutogradContext* ctx, std::vector<torch::Tensor> grad_output)
    {
        std::vector<torch::Tensor> input = ctx->get_saved_variables();
        auto multi_grid                  = input[0];
        auto index                       = input[1];
        auto uv                          = input[2];

        CHECK_EQ(grad_output.size(), 1);

        auto grad_multi_grid = torch::zeros_like(multi_grid);
        auto grad_uv         = torch::zeros_like(uv);

        grad_multi_grid.set_requires_grad(1);
        grad_uv.set_requires_grad(1);

        // if (multi_grid.is_cpu())
        // {
        //     IndirectGridSample3DBackwardCPU(multi_grid, index, uv, grad_multi_grid, grad_uv, grad_output[0]);
        // }
        // else
        if (multi_grid.is_cuda())
        {
            IndirectGridSample2DBackwardCUDA(multi_grid, index, uv, grad_multi_grid, grad_uv, grad_output[0]);
            CUDA_SYNC_CHECK_ERROR();
        }
        else
        {
            CHECK(false);
        }
        // std::cout << "backward output" << std::endl;
        // PrintTensorInfo(grad_multi_grid);
        // PrintTensorInfo(grad_uv);

        return {grad_multi_grid, torch::Tensor(), grad_uv};
    }
} // namespace torch::autograd

torch::Tensor IndirectGridSample2D(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv)
{
    CHECK(multi_grid.defined());
    CHECK(index.defined());
    CHECK(uv.defined());

    CHECK_EQ(multi_grid.dim(), 4);
    CHECK_EQ(index.dim(), 1);
    CHECK_EQ(uv.dim(), 2);
    auto result = torch::autograd::IndirectGridSample2D::apply(multi_grid, index, uv);
    CHECK_EQ(result.size(), 1);
    return result.front();
}
