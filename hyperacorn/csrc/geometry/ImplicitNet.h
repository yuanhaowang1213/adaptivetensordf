#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "Settings.h"
#include "data/SceneBase.h"

#include "Modules.h"
#include "PositionalEncoding.h"


class ExternalFeaturesImpl : public torch::nn::Module
{
   public:
    ExternalFeaturesImpl(int num_nodes, int num_external_features)
    {
        features = torch::rand({num_nodes, num_external_features}) * 2 - 1;
        register_parameter("features", features);
    }

    torch::Tensor forward(torch::Tensor node_ids)
    {
        auto factor_selected = torch::index_select(features, 0, node_ids);
        return factor_selected;
    }
    torch::Tensor features;
};
TORCH_MODULE(ExternalFeatures);

class GridBiasImpl : public torch::nn::Module
{
   public:
    GridBiasImpl(int num_nodes, int num_grid_features, int dim)
    {
        if (dim == 3)
        {
            factor = torch::ones({num_nodes, num_grid_features, 1, 1, 1});
            bias   = torch::zeros({num_nodes, num_grid_features, 1, 1, 1});
        }
        else
        {
            CHECK(false);
        }
        register_parameter("factor", factor);
        register_parameter("bias", bias);
    }

    torch::Tensor forward(torch::Tensor grid, torch::Tensor node_ids)
    {
        auto factor_selected = torch::index_select(factor, 0, node_ids);
        auto bias_selected   = torch::index_select(bias, 0, node_ids);
        grid                 = grid * factor_selected;
        grid                 = grid + bias_selected;
        return grid;
    }
    torch::Tensor factor;
    torch::Tensor bias;
};
TORCH_MODULE(GridBias);



// Stores a feature grid for every node explicitly in memory.
// The forward function then copies the feature grid into an output tensor based on the
// the node index (index_select).
class ExplicitFeatureGridImpl : public torch::nn::Module
{
   public:
    ExplicitFeatureGridImpl(int num_nodes, int out_features, std::vector<long> output_grid,  std::string init = "zero", float init_scale = 0.5f)
    {
        init1 = init;
        init_scale1 = init_scale;
        sizes.push_back(num_nodes);
        sizes.push_back(out_features);
        for (auto g : output_grid)
        {
            sizes.push_back(g);
        }
        std::cout << "Explicit Feature Grid \n" << std::endl;
        std::cout << sizes << std::endl;
        if (init1 == "uniform" || init1 == "random")
        {
            grid_data = torch::empty(sizes);
            grid_data.uniform_(-init_scale1, init_scale1);
        }
        else if (init1 == "minus")
        {
            grid_data = -torch::ones(sizes);
        }
        else if (init1 == "zero")
        {
            grid_data = torch::zeros(sizes);
        }
        else
        {
            CHECK(false) << "Unknown grid init: " << init1 << ". Expected: uniform, minus, zero";
        }
        register_parameter("grid_data", grid_data);
        grid_tmp = grid_data;
    }
    void reset()
    {
        if (init1 == "uniform" || init1 == "random")
        {

            torch::nn::init::uniform_(grid_data,-init_scale1, init_scale1 );
            // register_parameter("grid_data", grid_data);
        }
        else if (init1 == "minus")
        {
            torch::nn::init::ones_(grid_data);
        }
        else if (init1 == "zero")
        {
            torch::nn::init::zeros_(grid_data);
        }
        else
        {
            CHECK(false) << "Unknown grid init: " << init1 << ". Expected: uniform, minus, zero";
        }

    }

    at::Tensor forward(at::Tensor node_index)
    {
        // CHECK(torch::equal(grid_tmp, grid_data));
        auto feature_grid = torch::index_select(grid_data, 0, node_index);
        return feature_grid;
    }

    torch::Tensor grid_data;
    std::string init1;
    torch::Tensor grid_tmp;
    std::vector<long> sizes;
    float init_scale1;
};
TORCH_MODULE(ExplicitFeatureGrid);

// Takes a vector and generates a grid.
//
// Example:
// With input_features=2, output_grid=(16,16), out_features=8
// [n, 2] -> [n, 8, 16, 16]
//
class ImplicitFeatureGridImpl : public torch::nn::Module
{
   public:
    ImplicitFeatureGridImpl(int in_features, int external_features, int out_features, std::vector<long> output_grid,
                            bool use_pe, int num_encoding_functions, int num_hidden_layers, int hidden_features,
                            std::string activation)
        : in_features(in_features), out_features(out_features), output_grid(output_grid)
    {
        std::cout << "Grid Generator" << std::endl;
        int features_after_pe = in_features + external_features;
        if (use_pe)
        {
            positional_encoding = PositionalEncoding(num_encoding_functions);
            features_after_pe   = positional_encoding->OutputFeatures(in_features) + external_features;
            // std::cout << "Positional Encoding n=" << num_encoding_functions << std::endl;
            // std::cout << "external_features = " << external_features << std::endl;
            register_module("positional_encoding", positional_encoding);
        }
        std::cout << in_features << " (PE) ";

        int total_feature_grid_size = out_features;
        for (auto i : output_grid) total_feature_grid_size *= i;

        coord2features_net =
            FCBlock(features_after_pe, total_feature_grid_size, num_hidden_layers, hidden_features, true, activation);


        register_module("coord2features_net", coord2features_net);
    }


    torch::Tensor forward(torch::Tensor coords, torch::Tensor external_features)
    {
        CHECK_EQ(coords.dim(), 2);
        CHECK_EQ(coords.size(-1), in_features);
        if (positional_encoding)
        {
            // float [num_coords, num encoded features]
            coords = positional_encoding->forward(coords);
        }

        if (external_features.defined())
        {
            CHECK_EQ(coords.size(0), external_features.size(0));
            coords = torch::cat({coords, external_features}, 1);
        }


        auto features = coord2features_net->forward(coords);
        CHECK_EQ(features.dim(), 2);

        std::vector<long> output_shape;
        output_shape.push_back(features.size(0));
        output_shape.push_back(out_features);
        for (auto o : output_grid)
        {
            output_shape.push_back(o);
        }

        auto feature_grid = features.squeeze().reshape(output_shape);
        return feature_grid;
    }

    int in_features;
    int out_features;
    std::vector<long> output_grid;

    PositionalEncoding positional_encoding = nullptr;
    FCBlock coord2features_net             = nullptr;
};
TORCH_MODULE(ImplicitFeatureGrid);

class NeuralGridSamplerImpl : public torch::nn::Module
{
   public:
    NeuralGridSamplerImpl(bool swap_xy, bool align_corners) : swap_xy(swap_xy), align_corners(align_corners) {}

    torch::Tensor forward(at::Tensor features_in, at::Tensor relative_coordinates)
    {
        CHECK_EQ(relative_coordinates.dim(), 3);
        int D = relative_coordinates.size(2);
        CHECK_EQ(features_in.dim(), D + 2);

        auto opt = torch::nn::functional::GridSampleFuncOptions();
        opt      = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(align_corners);

        // opt      = opt.padding_mode(torch::kReflection).mode(torch::kBilinear).align_corners(align_corners);

        // Note: grid_sample has xy indexing. The tree and everything has yx indexing.
        // -> swap coordinates for grid_sample
        if (D == 2)
        {
            if (swap_xy)
            {
                relative_coordinates =
                    torch::cat({relative_coordinates.slice(2, 1, 2), relative_coordinates.slice(2, 0, 1)}, 2);
            }
            relative_coordinates = relative_coordinates.unsqueeze(1);
        }
        else if (D == 3)
        {
            if (swap_xy)
            {
                relative_coordinates =
                    torch::cat({relative_coordinates.slice(2, 2, 3), relative_coordinates.slice(2, 1, 2),
                                relative_coordinates.slice(2, 0, 1)},
                               2);
            }
            relative_coordinates = relative_coordinates.unsqueeze(1).unsqueeze(1);
        }
        else
        {
            CHECK(false);
        }
        // 3D: [batches, num_features, 1, 1, batch_size]
        auto neural_samples = torch::nn::functional::grid_sample(features_in, relative_coordinates, opt);

        // After squeeze:
        // [batches, num_features, batch_size]
        if (D == 2)
        {
            neural_samples = neural_samples.squeeze(2);
        }
        else if (D == 3)
        {
            neural_samples = neural_samples.squeeze(2).squeeze(2);
        }
        else
        {
            CHECK(false);
        }

        // [batches, batch_size, num_features]
        neural_samples = neural_samples.permute({0, 2, 1});

        return neural_samples;
    }

    bool swap_xy;
    bool align_corners;
};
TORCH_MODULE(NeuralGridSampler);
