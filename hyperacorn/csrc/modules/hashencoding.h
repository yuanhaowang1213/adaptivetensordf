#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"

#include <torch/torch.h>

namespace torch::autograd
{
    struct HashEncoding : public Function<HashEncoding>
    {
        static std::vector<torch::Tensor> forward(AutogradContext * ctx, torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
                                                torch::Tensor variables);
                                                // int per_level_scale, int base_resolution, int calc_grad_inputs, int grid_type, int align_corners);
        static std::vector<torch::Tensor> backward(AutogradContext * ctx, std::vector<torch::Tensor> grad_output);
    };
}

torch::Tensor HashEncoding(torch::Tensor inputs, torch::Tensor embeddings, torch::Tensor offsets,
                            float per_level_scale, int base_resolution, int calc_grad_inputs, int grid_type, int align_corners);
