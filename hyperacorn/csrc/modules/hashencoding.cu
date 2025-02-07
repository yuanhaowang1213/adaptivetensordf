#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"
#include <ATen/cuda/CUDAContext.h>
#include <ATen/autocast_mode.h>

#include "hashencoding.h"
#include "Settings.h"
#include <torch/autograd.h>

#include <torch/csrc/autograd/custom_function.h>

#include "gridencoder.h"
using namespace Saiga;

#undef MIN
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#undef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// void HashEncodingForwardCPU(torch::Tensor inputs, torch::Tensor embeddings , torch::Tensor offsets, int per_level_scale,
//                             int B, int D, int C, int L, int S, int H, torch::Tensor dy_dx, int gridtype, int align_corners, torch::Tensor& output)
// {

// }




namespace torch::autograd
{
    std::vector<torch::Tensor> HashEncoding::forward(AutogradContext * ctx, torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
                                        torch::Tensor variables)
                                        // int per_level_scale, int base_resolution, int calc_grad_inputs, int grid_type, int align_corners)
    {


        // printf("test autograd here 0\n");
        uint32_t B = inputs.size(0);
        uint32_t D = inputs.size(1);
        uint32_t L = offsets.size(0) - 1;
        uint32_t C = embeddings.size(1);

        // printf("test autograd here 0.1\n");

        // PrintTensorInfo(variables);

        auto variable_test = variables.to(torch::kCPU);
        float per_level_scale = variable_test.data_ptr<float>()[0 * variables.stride(0)];
        int base_resolution = (int)variable_test.data_ptr<float>()[1 * variables.stride(0)];
        int calc_grad_inputs = (int)variable_test.data_ptr<float>()[2 * variables.stride(0)];
        uint32_t grid_type       = (int)variable_test.data_ptr<float>()[3 * variables.stride(0)];
        int align_corners   = (int)variable_test.data_ptr<float>()[4 * variables.stride(0)];

        // printf("test autograd here 0.2\n");

        float S = std::log2((float)per_level_scale);
        uint32_t H = base_resolution;

        bool align_c;
        if(align_corners > 0)
        {
            align_c = true;
        }
        else
        {
            align_c = false;
        }

        // printf("C is %d \n", C);
        // std::cout <<"autocast enable " << at::autocast::is_enabled() <<"calc grad " <<calc_grad_inputs<< "c/2" << C%2 <<std::endl;
        if(at::autocast::is_enabled() && C%2 == 0)
        {
            embeddings = embeddings.to(torch::kHalf);
            // printf("turning in to half\n");
        }
        // # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        // # if C % 2 != 0, force float, since half for atomicAdd is very slow.
        // if torch.is_autocast_enabled() and C % 2 == 0:
        //     embeddings = embeddings.to(torch.half)
        // torch::Tensor outputs = torch::zeros({L, B, C}, torch::TensorOptions().dtype(embeddings.dtype()).device(embeddings.device()) );
        // torch::TensorOptions(torch::kFloat).device(sample_values.device())
        torch::Tensor outputs = torch::empty({L,B,C}, torch::TensorOptions(embeddings.dtype()).device(embeddings.device()));

        torch::Tensor dy_dx;
        if (calc_grad_inputs > 0)
        {
            dy_dx = torch::empty({B, L*D*C}, torch::TensorOptions(embeddings.dtype()).device(embeddings.device()));
        }
        // std::cout << "dy dx test " << " " << dy_dx.defined() << std::endl;
        // printf("has encoding test 0.001\n");
        // std::cout <<"calc grad " << calc_grad_inputs << std::endl;
        // PrintTensorInfo(dy_dx);
        // std::cout <<"scalar type " << inputs.scalar_type() <<" " <<embeddings.scalar_type() << " " << offsets.scalar_type() <<" " << outputs.scalar_type() <<" " <<dy_dx.scalar_type() << std::endl;
        // PrintTensorInfo(inputs);
        // PrintTensorInfo(embeddings);
        // PrintTensorInfo(outputs);
        // PrintTensorInfo(offsets);
        // PrintTensorInfo(dy_dx);
        grid_encode_forward(inputs, embeddings, offsets, outputs, B, D, C, L, S, H, dy_dx, grid_type, align_c );

        // PrintTensorInfo(outputs);
        outputs = torch::reshape(torch::permute(outputs, {1,0,2}), {B, L*C});

        // printf("outputs shape \n");
        // PrintTensorInfo(outputs);
        // void grid_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets,
        //     at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S,
        //     const uint32_t H, at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners) {

        // printf("hash encoding test 0.002\n");
        // grid_encode_forward( const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H,
        //      at::optional<at::Tensor> dy_dx, const uint32_t gridtype, const bool align_corners);
        std::vector<torch::Tensor> input_variables;
        input_variables.push_back(inputs);
        input_variables.push_back(embeddings);
        input_variables.push_back(offsets);
        input_variables.push_back(dy_dx);
        ctx->save_for_backward(input_variables);

        // std::vector<float> scale_variables;
        // scale_variables.push_back((float)B);
        // scale_variables.push_back((float)D);
        // scale_variables.push_back((float)C);
        // scale_variables.push_back((float)L);
        // scale_variables.push_back((float)S);
        // scale_variables.push_back((float)H);
        // scale_variables.push_back((float)grid_type);
        // scale_variables.push_back((float)align_corners);

        // ctx->saved_data["B"] = (int)B;
        // ctx->saved_data["D"] = (int)D;
        // ctx->saved_data["C"] = (int)C;
        // ctx->saved_data["L"] = (int)L;
        // ctx->saved_data["S"] = S;
        // ctx->saved_data["H"] = (int)H;
        // ctx->saved_data["grid_type"] = (int)grid_type;
        ctx->saved_data["variables"] = variables;

        std::vector<torch::Tensor> result;
        result.push_back(outputs);
        return result;

    }
    // std::vector<torch::Tensor> HashEncoding::backward(AutogradContext * ctx, std::vector<torch::Tensor> grad_output)
    std::vector<torch::Tensor>  HashEncoding::backward(AutogradContext * ctx, std::vector<torch::Tensor>  grad_output)
    {
        std::vector<torch::Tensor> input_variables = ctx->get_saved_variables();
        auto inputs         = input_variables[0];
        auto embeddings     = input_variables[1];
        auto offsets        = input_variables[2];
        auto dy_dx          = input_variables[3];

        auto variables      = ctx->saved_data["variables"].toTensor();
        uint32_t B = inputs.size(0);
        uint32_t D = inputs.size(1);
        uint32_t L = offsets.size(0) - 1;
        uint32_t C = embeddings.size(1);

        auto variable_test = variables.to(torch::kCPU);

        float per_level_scale = variable_test.data_ptr<float>()[0 * variables.stride(0)];
        int base_resolution = variable_test.data_ptr<float>()[1 * variables.stride(0)];
        int calc_grad_inputs = variable_test.data_ptr<float>()[2 * variables.stride(0)];
        int grid_type       = variable_test.data_ptr<float>()[3 * variables.stride(0)];
        int align_corners   = variable_test.data_ptr<float>()[4 * variables.stride(0)];

        float S = std::log2((float)per_level_scale);
        uint32_t H = base_resolution;

        bool align_c;
        if(align_corners > 0)
        {
            align_c = true;
        }
        else
        {
            align_c = false;
        }

        // // std::vector<float> scale_variables = ctx->saved_data["scale_variables"];
        // uint32_t B               = ctx->saved_data["B"].toInt();
        // uint32_t D               = ctx->saved_data["D"].toInt();
        // uint32_t C               = ctx->saved_data["C"].toInt();
        // uint32_t L               = ctx->saved_data["L"].toInt();
        // float S                  = ctx->saved_data["S"].toDouble();
        // uint32_t H               = ctx->saved_data["H"].toInt();;
        // uint32_t gridtype        = ctx->saved_data["gridtype"].toInt();
        // uint32_t align_corners    = ctx->saved_data["align_corners"].toInt();

        // auto grad = torch::permute(torch::reshape(grad_output[0], {B,L,C}),{1,0,2});
        // auto grad = torch::permute(torch::reshape(grad_output[0], {B,L,C}),{1,0,2}).contiguous();
        auto grad_embeddings = torch::zeros_like(embeddings);

        torch::Tensor grad_inputs;
        if(dy_dx.defined())
        {
            grad_inputs = torch::zeros_like(inputs, embeddings.options());
        }
        // grid_encode_backward(grad, inputs, embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, grid_type, align_c);
        // printf("test backward\n");
        grid_encode_backward(torch::permute(torch::reshape(grad_output[0], {B,L,C}),{1,0,2}).contiguous(), inputs,
                    embeddings, offsets, grad_embeddings, B, D, C, L, S, H, dy_dx, grad_inputs, grid_type, align_c);

        if(dy_dx.defined())
        {
            grad_inputs = grad_inputs.toType(inputs.scalar_type());
        }

        return {grad_inputs, grad_embeddings,torch::Tensor(), torch::Tensor()};
    }
}

torch::Tensor HashEncoding(torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
                            float per_level_scale, int base_resolution, int calc_grad_inputs, int grid_type, int align_corners)
{
    CHECK(inputs.defined());
    CHECK(embeddings.defined());
    CHECK(offsets.defined());
    CHECK_EQ(inputs.dim(),2);

    // printf("test hash encoding func 0\n");
    std::vector<float> variable_vec;
    variable_vec.push_back((float)per_level_scale);
    variable_vec.push_back((float)base_resolution);
    variable_vec.push_back((float)calc_grad_inputs);
    variable_vec.push_back((float)grid_type);
    variable_vec.push_back((float)align_corners);
    // std::cout <<"input test " << per_level_scale <<" " <<base_resolution <<" " << calc_grad_inputs <<" " <<grid_type <<" " <<align_corners << std::endl;

    torch::Tensor variables = torch::from_blob(&variable_vec[0], {(long)variable_vec.size()}, torch::TensorOptions().dtype(torch::kFloat32))
                                           .clone()
                                           .to(device);
;

    // std::cout <<"input device " << inputs.device() << std::endl;
    CHECK_EQ(inputs.device(), offsets.device());
    CHECK_EQ(inputs.device(), embeddings.device());
    CHECK_EQ(inputs.device(), variables.device());

    // printf("test before hash encoding\n");
    auto result = torch::autograd::HashEncoding::apply(inputs, embeddings, offsets, variables);
    // auto result = torch::autograd::HashEncoding::apply(inputs, embeddings, offsets, per_level_scale,
    //                                         base_resolution, calc_grad_inputs, grid_type, align_corners);
    // auto result = torch::autograd::IndirectGridSample3D::apply(multi_grid, index, uv);

    CHECK_EQ(result.size(), 1);
    return result.front();
}

// torch::Tensor HashEncoding(torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
//     int per_level_scale, int base_resolution, int calc_grad_inputs, int grid_type, int align_corners);
