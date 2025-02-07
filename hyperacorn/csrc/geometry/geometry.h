#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/assert.h"
#include "saiga/cuda/imgui_cuda.h"

#include "ImplicitNet.h"
#include "Settings.h"
#include "data/SceneBase.h"

// #include "fourierlayer.h"
// #include "dncnn.h"
#include "saiga/core/image/freeimage.h"
#include "saiga/vision/torch/ColorizeTensor.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/ImageTensor.h"



#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

#include "randomfourierfeature.h"

using namespace Saiga;




class TVLoss
{
   public:
    // TV Loss for N-dimensional input.
    //
    // Input
    //      grid [num_batches, num_channels, x, y, z, ...]
    //      weight [num_batches]
    torch::Tensor forward(torch::Tensor grid, torch::Tensor weight = {})
    {
        int num_channels = grid.size(1);
        int D            = grid.dim() - 2;
        // int num_batches  = grid.size(0);
        // int num_channels = grid.size(1);


        torch::Tensor total_loss;
        for (int i = 0; i < D; ++i)
        {
            int d     = i + 2;
            int size  = grid.size(d);
            torch::Tensor loss;
            if(D == 1)
            {
                loss = (grid.slice(d, 0, size - 1) - grid.slice(d, 1, size)).abs().mean({1, 2});
            }
            else if(D == 2)
            {
                loss = (grid.slice(d, 0, size - 1) - grid.slice(d, 1, size)).abs().mean({1, 2, 3});
            }
            else if(D == 3)
            {
                loss = (grid.slice(d, 0, size - 1) - grid.slice(d, 1, size)).abs().mean({1, 2, 3, 4});
            }
            else
            {
                printf("only applied in 1D 2D and 3D");
            }

            if (weight.defined())
            {
                loss *= weight;
            }

            loss = loss.mean();

            if (total_loss.defined())
            {
                total_loss += loss;
            }
            else
            {
                total_loss = loss;
            }
        }
        CHECK(total_loss.defined());

        return total_loss * num_channels;
    }

    torch::Tensor forward_huber(torch::Tensor grid, float huber_factor)
    {
        int num_channels = grid.size(1);
        int D            = grid.dim() - 2;
        // int num_batches  = grid.size(0);
        // int num_channels = grid.size(1);


        torch::Tensor total_loss;
        for (int i = 0; i < D; ++i)
        {
            int d     = i + 2;
            int size  = grid.size(d);
            torch::Tensor loss;
            
                // loss = (grid.slice(d, 0, size - 1) - grid.slice(d, 1, size)).abs().mean({1, 2});
            loss = torch::nn::functional::huber_loss(grid.slice(d, 0, size - 1), grid.slice(d, 1, size), torch::nn::functional::HuberLossFuncOptions().reduction(torch::kMean).delta(huber_factor));
            // PrintTensorInfo(grid);
            // PrintTensorInfo(loss);

            if (total_loss.defined())
            {
                total_loss += loss;
            }
            else
            {
                total_loss = loss;
            }
        }
        CHECK(total_loss.defined());

        return total_loss * num_channels;
    }

};


class TVLoss_global
{
    public:
    // Input grid[x,y,z]
    torch::Tensor forward(torch::Tensor grid)
    {
        torch::Tensor total_loss;
        int D = grid.dim();
        for(int i = 0; i < D;++i)
        {
            int size = grid.size(i);
            auto loss = (grid.slice(i, 0, size-1) - grid.slice(i, 1, size)).abs().mean();
            if(total_loss.defined())
            {
                total_loss += loss;
            }
            else
            {
                total_loss = loss;
            }
        }
        CHECK(total_loss.defined());
        return total_loss;

    }
};




class NeuralGeometry : public torch::nn::Module
{
   public:
    NeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params)
        : num_channels(num_channels), D(D), params(params)
    {
    }

    virtual void train(int epoch_id, bool on)
    {
        torch::nn::Module::train(on);
        c10::cuda::CUDACachingAllocator::emptyCache();
        if (on)
        {
            if (!optimizer_adam && !optimizer_sgd)
            {
                CreateGeometryOptimizer();
            }
            if (optimizer_adam) optimizer_adam->zero_grad();
            if (optimizer_sgd) optimizer_sgd->zero_grad();
            if (optimizer_rms) optimizer_rms->zero_grad();
            if (optimizer_decoder) optimizer_decoder->zero_grad();
        }
    }

    void ResetGeometryOptimizer() { CreateGeometryOptimizer(); }

    void CreateGeometryOptimizer()
    {
        optimizer_adam =
            std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions().lr(10));

        optimizer_rms = std::make_shared<torch::optim::RMSprop>(std::vector<torch::Tensor>(),
                                                                torch::optim::RMSpropOptions().lr(10));

        optimizer_sgd = std::make_shared<torch::optim::SGD>(std::vector<torch::Tensor>(), torch::optim::SGDOptions(10));
        AddParametersToOptimizer();
    }



    virtual void PrintInfo() {}
    virtual void PrintGradInfo(int epoch_id, TensorBoardLogger* logger) {}


    void OptimizerStep(int epoch_id)
    {
        if (optimizer_sgd)
        {
            optimizer_sgd->step();
            optimizer_sgd->zero_grad();
        }
        if (optimizer_adam)
        {
            optimizer_adam->step();
            optimizer_adam->zero_grad();
        }
        if (optimizer_rms)
        {
            optimizer_rms->step();
            optimizer_rms->zero_grad();
        }
        if (optimizer_decoder)
        {
            optimizer_decoder->step();
            optimizer_decoder->zero_grad();
        }
    }

    void UpdateLearningRate(double factor)
    {
        if (optimizer_adam) UpdateLR(optimizer_adam.get(), factor);
        if (optimizer_sgd) UpdateLR(optimizer_sgd.get(), factor);
        if (optimizer_rms) UpdateLR(optimizer_rms.get(), factor);
        if (optimizer_decoder) UpdateLR(optimizer_decoder.get(), factor);
    }

    void UpdateDecoderLearningRate(double factor)
    {
        if(optimizer_decoder)  UpdateLR(optimizer_decoder.get(), factor);
    }

    std::vector<double> getDecoderLearningRate()
    {
        // auto optimizer_lr = optimizer_decoder.get();
        std::vector<double> learning_rate;
        for (auto& pg : optimizer_decoder.get()->param_groups())
        {
            auto opt_rms = dynamic_cast<torch::optim::RMSpropOptions*>(&pg.options());
            if (opt_rms)
            {
                // opt_rms->lr() = opt_rms->lr() * factor;
                learning_rate.push_back(opt_rms->lr());
            }
        }

        return learning_rate;
    }
    // Compute the 'simple' integral by just adding each sample value to the given ray index.
    // The ordering of the samples is not considered.
    //
    // Computes:
    //      sample_integral[ray_index[i]] += sample_value[i]
    //
    // Input:
    //      sample_value [num_groups, group_size, num_channels]
    //      ray_index [N]
    //
    // Output:
    //      sample_integral [num_channels, num_rays]
    //
    torch::Tensor IntegrateSamplesXRay(torch::Tensor sample_values, torch::Tensor integration_weight,
                                       torch::Tensor ray_index, int num_channels, int num_rays);


    // Blends the samples front-to-back using alpha blending. This is used for a RGB-camera model (non xray) and the
    // implementation follows the raw2outputs function of NeRF. However, in our case it is more complicated because
    // each ray can have a different number of samples. The computation is done in the following steps:
    //
    //  1. Sort the sample_values into a matrix of shape: [num_rays, max_samples_per_ray, num_channels]
    //     Each row, is also ordered correctly in a front to back fashion. If a ray has less than max_samples_per_ray
    //     samples, the remaining elements are filled with zero.
    //
    //
    // Input:
    //      sample_value [any_shape, num_channels]
    //      ray_index [any_shape]
    //
    //      // The local id of each sample in the ray. This is used for sorting!
    //      sample_index_in_ray [any_shape]
    //
    // Output:
    //      sample_integral [num_channels, num_rays]
    //
    torch::Tensor IntegrateSamplesAlphaBlending(torch::Tensor sample_values, torch::Tensor integration_weight,
                                                torch::Tensor ray_index, torch::Tensor sample_index_in_ray,
                                                int num_channels, int num_rays, int max_samples_per_ray);


    torch::Tensor Filter2dIndependentChannels(torch::Tensor x, Matrix<float, -1, -1> kernel, int padding)
    {
        SAIGA_ASSERT(x.dim() == 4);
        torch::Tensor K = FilterTensor(kernel);
        K               = K.repeat({x.size(1), 1, 1, 1}).to(x.device());
        auto res        = torch::conv2d(x, K, {}, 1, padding, 1, x.size(1));
        return res;
    }
   protected:
    std::shared_ptr<torch::optim::Adam> optimizer_decoder;

    std::shared_ptr<torch::optim::Adam> optimizer_adam;
    std::shared_ptr<torch::optim::SGD> optimizer_sgd;
    std::shared_ptr<torch::optim::RMSprop> optimizer_rms;

    int num_channels;
    int D;
    std::shared_ptr<CombinedParams> params;

    virtual void AddParametersToOptimizer() = 0;
};

class HierarchicalNeuralGeometry : public NeuralGeometry
{
   public:
    HierarchicalNeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params, HyperTreeBase tree);

    std::pair<torch::Tensor, torch::Tensor> AccumulateSampleLossPerNode(const NodeBatchedSamples& combined_samples,
                                                                        torch::Tensor per_ray_loss);
    std::pair<torch::Tensor, torch::Tensor> AccumulateSampleLossPerNode(const SampleList& combined_samples,
                                                                        torch::Tensor per_ray_loss);

    virtual torch::Tensor VolumeRegularizer() { return torch::Tensor(); }

    virtual void SampleVolumeTest(std::string output_vol_file) {}

    HyperTreeBase tree = nullptr;

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> ComputeImage(SampleList all_samples, int num_channels, int num_pixels, bool use_decoder );

    virtual void reset_vgg_parameters() {};

    virtual void Compute_edge_nlm_samples(bool scale0_start) {};

    virtual torch::Tensor Testcode(std::string ep_dir) { return torch::Tensor();};

    virtual void SaveTensor(std::string ep_dir) {};
    // Input:
    //   global_coordinate [num_samples, D]
    //   node_id [num_samples]
    virtual torch::Tensor SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id) { return {}; }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> IntegrateSamplesXRay_Bound(torch::Tensor sample_values, torch::Tensor integration_weight, torch::Tensor ray_index,
                                                        torch::Tensor weight_bound_index_inv, int num_channels, int num_rays);
    // Output:
    //      value [num_groups, group_size, num_channels]
    torch::Tensor SampleVolumeBatched(torch::Tensor global_coordinate, torch::Tensor sample_mask, torch::Tensor node_id, bool use_decoder = true)
    {
        CHECK_EQ(global_coordinate.dim(), 3);
        if (global_coordinate.numel() == 0)
        {
            return torch::empty({global_coordinate.size(0), global_coordinate.size(1), num_channels},
                                global_coordinate.options());
        }

        // {
        //     auto local_samples = tree->ComputeLocalSamples(global_coordinate, node_id);
        //     local_samples = local_samples * sample_mask;
        //     if(local_samples.abs().min().item().toFloat() > 0 || local_samples.abs().max().item().toFloat()>1)
        //     {
        //         printf("there is an error\n");
        //         PrintTensorInfo(local_samples);
        //     }
        // }
        torch::Tensor neural_features, density;
        {
            neural_features = SampleVolume(global_coordinate, node_id);
        }
        // if(0)
        // {
        //     if(params->net_params.using_fourier)
        //     {
        //         SAIGA_OPTIONAL_TIME_MEASURE("FourierProcess", timer);

        //         neural_features = FourierProcess(global_coordinate, neural_features);
        //         // printf("after fourier features\n");
        //         // PrintTensorInfo(neural_features);
        //     }
        // }
        // printf("after sample volume\n");
        // PrintTensorInfo(neural_features);
        if(use_decoder || params->net_params.using_tensor_decoder)
        {
            SAIGA_OPTIONAL_TIME_MEASURE("DecodeFeatures", timer);
            density = DecodeFeatures(neural_features);
            // std::cout << "get density " << std::endl;
        }
        else
        {
            CHECK_EQ(density.sizes()[-1], 1);
            if (params->net_params.last_activation_function == "relu")
            {
                density = torch::relu(neural_features);
            }
            else if (params->net_params.last_activation_function == "abs")
            {
                density = torch::abs(neural_features);
            }
            else if (params->net_params.last_activation_function == "softplus")
            {
                density = torch::softplus(neural_features, params->net_params.softplus_beta);
            }
            else if (params->net_params.last_activation_function == "id")
            {
            }
            else
            {
                CHECK(false);
            }
        }
        // printf("test samplevolumebached\n");
        // PrintTensorInfo(density);
        // PrintTensorInfo(sample_mask);
        CHECK_EQ(density.dim(), sample_mask.dim());
        density = density * sample_mask;
        // PrintTensorInfo(density);
        // printf("end test\n");
        return density;
    }
    bool in_roi(float * node_min_ptr, float * node_max_ptr, float * node_mid_ptr, vec3 roi_min, vec3 roi_max, int node_id, float epsilon)
    {
        float roi_min_y = roi_min[1] - epsilon;
        float roi_max_y = roi_max[1] + epsilon;

        float roi_min_x = roi_min[0] ;
        float roi_max_x = roi_max[0] ;
        float roi_min_z = roi_min[2] ;
        float roi_max_z = roi_max[2] ;

        if( ((node_min_ptr[node_id * 3] > roi_min_x && node_min_ptr[node_id * 3] < roi_max_x )||
            (node_max_ptr[node_id * 3] > roi_min_x && node_max_ptr[node_id * 3] < roi_max_x )||
            (node_mid_ptr[node_id * 3] > roi_min_x && node_mid_ptr[node_id * 3] < roi_max_x )) &&
            ((node_min_ptr[node_id * 3 + 1] > roi_min_y && node_min_ptr[node_id * 3 + 1] < roi_max_y) ||
            (node_max_ptr[node_id * 3 + 1] > roi_min_y && node_max_ptr[node_id * 3 + 1] < roi_max_y) ||
            (node_mid_ptr[node_id * 3 + 1] > roi_min_y && node_mid_ptr[node_id * 3 + 1] < roi_max_y)) &&
            ((node_min_ptr[node_id * 3 + 2] > roi_min_z && node_min_ptr[node_id * 3 + 2] < roi_max_z) ||
            (node_max_ptr[node_id * 3 + 2] > roi_min_z && node_max_ptr[node_id * 3 + 2] < roi_max_z) ||
            (node_mid_ptr[node_id * 3 + 2] > roi_min_z && node_mid_ptr[node_id * 3 + 2] < roi_max_z))  )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    std::tuple<torch::Tensor, Eigen::Vector<int, -1>> GenerateROI_global(std::vector<long> shape, vec3 roi_min, vec3 roi_max)
    {
        Eigen::Vector<int, -1> shape_v;
        Eigen::Vector<float, -1> roi_min_v;
        Eigen::Vector<float, -1> step_size_v;

        std::vector<long> shape_new;

        float step = std::max(std::max((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[1] - roi_min[1])/(shape[1]-1)), (roi_max[2] - roi_min[2])/(shape[2]-1));

        for(int i = 0; i < shape.size(); ++i)
        {
            shape_new.push_back( int( std::round((roi_max[i] - roi_min[i])/step +1)) );
        }
        printf("shape_new is %ld %ld %ld\n", shape_new[0], shape_new[1], shape_new[2]);
        shape_v.resize(shape_new.size());
        roi_min_v.resize(roi_min_v.size());
        step_size_v.resize(roi_min.size());
        // roi_min_v.resize(3);
        // step_size_v.resize(3);
        for(int i = 0; i < shape_new.size(); ++i)
        {
            shape_v(i) = shape_new[i];
            roi_min_v(i) = roi_min[i];
            step_size_v(i) = step;
        }
        // Eigen::Vector<float, 3> roi_min_v (roi_min[0],roi_min[1],roi_min[2]);
        // Eigen::Vector<float, 3> step_size_v (step,step,step);
        torch::Tensor global_coordinate = tree->UniformPhantomSamplesGPUSlice_global( shape_v, false, roi_min_v, step_size_v);

        return {global_coordinate, shape_v};
    }

    virtual void to(torch::Device device, bool non_blocking = false) override
    {
        NeuralGeometry::to(device, non_blocking);
    }

    // Returns [volume_density, volume_node_index, volume_valid]
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> UniformSampledVolume(std::vector<long> shape,
                                                                                 int num_channels ,vec3 roi_min, vec3 roi_max, bool is_final );

    // void TestVolume();
    // slice_dim 0,1,2 for z,y,x
    void SaveVolume(TensorBoardLogger* tblogger, std::string tb_name, std::string out_dir, int num_channels,
                    float intensity_scale, int size, int slice_dim, vec3 roi_min, vec3 roi_max );

    static FCBlock shared_decoder;
    FCBlock decoder                        = nullptr;
    // FINN decoder                              = nullptr;

    // if(0)
    // {
    //     // static MultiscaleBacon shared_fourier_layer;
    //     MultiscaleBacon fourier_layer          = nullptr;
    // }
    // Evaluates the octree at the inactive-node's feature positions and sets the respective feature vectors.
    // This should be called before changing the octree structure, because then some inactive nodes will become active.
    // The newly active nodes will have a good initialization after this method.
    virtual void InterpolateInactiveNodes(HyperTreeBase old_tree) {}

    void setup_tv(torch::Tensor tv_losshere) {local_tv_loss = tv_losshere;};

    void setup_nlm(torch::Tensor nlm_loss) {local_nlm_loss = nlm_loss;};

    torch::Tensor local_nlm_loss;
    torch::Tensor local_tv_loss;

    torch::Tensor weight_bound_index;

   protected:
    // Takes the sample locations (and the corresponding tree-node-ids) and retrieves the values from the
    // hierarchical data structure. The input samples must be 'grouped' by the corresponding node-id.
    // The per-sample weight is multiplied to the raw sample output.
    //
    // Input:
    //      global_coordinate [num_groups, group_size, 3]
    //      weight            [num_groups, group_size]
    //      node_id           [num_groups]
    //
    // Output:
    //      value [num_groups, group_size, num_features]
    virtual torch::Tensor SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id) = 0;



    virtual void AddParametersToOptimizer();

    // Input:
    //      neural_features [num_groups, group_size, num_channels]
    virtual torch::Tensor DecodeFeatures(torch::Tensor neural_features);
    // if(0)
    // {
    //     torch::Tensor FourierProcess(torch::Tensor position, torch::Tensor neural_features);
    // }

   public:
    Saiga::CUDA::CudaTimerSystem* timer = nullptr;
};
