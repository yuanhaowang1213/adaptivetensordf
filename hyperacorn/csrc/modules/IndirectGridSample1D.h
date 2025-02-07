#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"

#include <torch/torch.h>

namespace torch::autograd
{
    struct IndirectGridSample1D : public Function<IndirectGridSample1D>
    {
        static std::vector<torch::Tensor> forward(AutogradContext* ctx, torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv);

        static std::vector<torch::Tensor> backward(AutogradContext* ctx, std::vector<torch::Tensor> grad_output);
    };
};

// Input: 
    // multi_grid: [nodes, features_num, y, x]
    // index : number of points
    // uv :    points
torch::Tensor IndirectGridSample1D(torch::Tensor multi_grid, torch::Tensor index, torch::Tensor uv);