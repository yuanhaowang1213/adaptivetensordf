#include "geometry.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "saiga/vision/torch/ColorizeTensor.h"


std::pair<torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::AccumulateSampleLossPerNode(
    const NodeBatchedSamples& combined_samples, torch::Tensor per_ray_loss)
{
    // [num_rays]
    per_ray_loss = torch::mean(per_ray_loss, 0).squeeze(0);

    auto linear_ray_index = combined_samples.ray_index.reshape({-1});
    auto per_sample_loss  = per_ray_loss.gather(0, linear_ray_index);
    per_sample_loss       = per_sample_loss.reshape(combined_samples.integration_weight.sizes());

    // [batches, batch_sample_per_node, channels]
    per_sample_loss = per_sample_loss * combined_samples.integration_weight;

    // project sample loss to nodes
    auto per_cell_loss   = torch::sum(per_sample_loss, {1, 2});
    auto per_cell_weight = torch::sum(combined_samples.integration_weight, {1, 2});

    auto per_cell_loss_sum   = torch::zeros({(long)tree->NumNodes()}, device);
    auto per_cell_weight_sum = torch::zeros({(long)tree->NumNodes()}, device);
    per_cell_loss_sum.scatter_add_(0, combined_samples.node_ids, per_cell_loss);
    per_cell_weight_sum.scatter_add_(0, combined_samples.node_ids, per_cell_weight);

    return {per_cell_loss_sum, per_cell_weight_sum};
}


std::pair<torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::AccumulateSampleLossPerNode(
    const SampleList& combined_samples, torch::Tensor per_ray_loss)
{
    // [num_rays]
    per_ray_loss = torch::mean(per_ray_loss, 0).squeeze(0);

    auto linear_ray_index = combined_samples.ray_index.reshape({-1});
    auto per_sample_loss  = per_ray_loss.gather(0, linear_ray_index);
    per_sample_loss       = per_sample_loss.reshape(combined_samples.weight.sizes());

    // [num_samples]
    per_sample_loss = per_sample_loss * combined_samples.weight;


    auto per_cell_loss_sum   = torch::zeros({(long)tree->NumNodes()}, device);
    auto per_cell_weight_sum = torch::zeros({(long)tree->NumNodes()}, device);
    per_cell_loss_sum.scatter_add_(0, combined_samples.node_id, per_sample_loss);
    per_cell_weight_sum.scatter_add_(0, combined_samples.node_id, combined_samples.weight);

    return {per_cell_loss_sum, per_cell_weight_sum};
}

FCBlock HierarchicalNeuralGeometry::shared_decoder = nullptr;

// if(0)
// {
//     MultiscaleBacon HierarchicalNeuralGeometry::shared_fourier_layer = nullptr;
// }
HierarchicalNeuralGeometry::HierarchicalNeuralGeometry(int num_channels, int D, std::shared_ptr<CombinedParams> params,
                                                       HyperTreeBase tree)
    : NeuralGeometry(num_channels, D, params), tree(tree)
{
    std::cout << ConsoleColor::BLUE << "=== Neural Geometry ===\n";

    int decoder_num_channels = num_channels;

    std::cout << "Last Activation: " << params->net_params.last_activation_function << std::endl;



    if (params->net_params.decoder_skip)
    {
        printf("skip decoder\n");
        params->net_params.grid_features = num_channels;
    }
    else
    {
        int features_after_pe;
        if( params->net_params.using_fourier)
        {
            features_after_pe = params->net_params.intermidiate_features;
        }
        else
        {
            features_after_pe = params->net_params.grid_features;
        }

        printf("grid feature test here \n");
        std::cout << features_after_pe << std::endl;
        // if (params->net_params.geometry_type == "exex")
        // {
        //     features_after_pe = features_after_pe * params->net_params.num_encoding_functions * 2 + features_after_pe;
        //     // in_features * vfrequency_bands.size() * 2 + in_features * include_input
        // }
        std::cout <<"using decoder " << params->net_params.using_decoder << "using tensor decoder " << params->net_params.using_tensor_decoder << std::endl;
        if(params->net_params.using_decoder && !params->net_params.using_tensor_decoder)
        {
            std::cout << "using original Decoder\n";
            decoder = FCBlock(features_after_pe, decoder_num_channels, params->net_params.decoder_hidden_layers,
                              params->net_params.decoder_hidden_features, true, params->net_params.decoder_activation);

            // decoder = FCBlock(3, decoder_num_channels, params->net_params.decoder_hidden_layers,
            //                   params->net_params.decoder_hidden_features, true, params->net_params.decoder_activation);
            // decoder = FINN(features_after_pe, decoder_num_channels, params->net_params.decoder_hidden_layers,
            //                   params->net_params.decoder_hidden_features, true, params->net_params.decoder_activation);

            register_module("decoder", decoder);
        }

        if ( params->net_params.using_decoder && !params->net_params.using_tensor_decoder)
        {
            std::cout <<"add original Decoder to the optimizer " << std::endl;
            optimizer_decoder =
                std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions().lr(10));
            std::cout << "Optimizing Decoder with LR " << params->net_params.decoder_lr << std::endl;

            auto options = std::make_unique<torch::optim::AdamOptions>(params->net_params.decoder_lr);
            // options->weight_decay(1e-2);
            // optimizer_adam->add_param_group({decoder->parameters(), std::move(options)});
            // if(params->net_params.using_decoder)
            {
                optimizer_decoder->add_param_group({decoder->parameters(), std::move(options)});
            }
        }
    }
    // if(0)
    // {
    //     if(params->net_params.using_fourier)
    //     {
    //         int in_features = params->net_params.grid_features;
    //         std::vector<float> fourier_freq, input_scale;
    //         for(int i = 0; i < params->net_params.fourier_frequency.size();++i)
    //         {
    //             fourier_freq.push_back(params->net_params.fourier_frequency(i));
    //         }


    //         // for(int i = 0; i <params->net_params.fourier_input_scale.size();++i )
    //         // {
    //         //     input_scale.push_back(params->net_params.fourier_input_scale(i));
    //         // }
    //         {
    //             std::cout << "using fourier features " << in_features << std::endl;
    //             fourier_layer = MultiscaleBacon(in_features, params->net_params.intermidiate_features, in_features, params->net_params.fourier_num_hidden_layers,  3,
    //                                             fourier_freq, input_scale );
    //             register_module("fourier_layer", fourier_layer);
    //         }
    //     }
    // }
}

// if(0)
// {
//     torch::Tensor HierarchicalNeuralGeometry::FourierProcess(torch::Tensor position, torch::Tensor neural_features)
//     {
//         torch::Tensor fourier_features;
//         if(fourier_layer)
//         {
//             CHECK(!shared_fourier_layer);
//             fourier_features = fourier_layer->forward(neural_features);
//         }
//         else if(shared_fourier_layer)
//         {
//             fourier_features = shared_fourier_layer->forward( neural_features);
//         }
//         else
//         {
//             fourier_features = neural_features;
//         }
//         return fourier_features;
//     }
// }

torch::Tensor HierarchicalNeuralGeometry::DecodeFeatures(torch::Tensor neural_features)
{
    //  [num_groups, group_size, num_channels]
    torch::Tensor decoded_features;
    // torch::Tensor tmp_no_use;
    if (decoder)
    {
        CHECK(!shared_decoder);
        // decoded_features = decoder->forward(neural_features, tmp_no_use);
        decoded_features = decoder->forward(neural_features);

    }
    else if (shared_decoder)
    {
        decoded_features = shared_decoder->forward(neural_features);
    }
    else
    {
        // CHECK(false);
        decoded_features = neural_features;
    }



    if (true)  // params->dataset_params.image_formation == "xray"
    {
        if (params->net_params.last_activation_function == "relu")
        {
            decoded_features = torch::relu(decoded_features);
        }
        else if (params->net_params.last_activation_function == "abs")
        {
            decoded_features = torch::abs(decoded_features);
        }
        else if (params->net_params.last_activation_function == "softplus")
        {
            decoded_features = torch::softplus(decoded_features, params->net_params.softplus_beta);
        }
        else if (params->net_params.last_activation_function == "id")
        {
        }
        else if (params->net_params.last_activation_function == "sigmoid")
        {
            decoded_features = torch::sigmoid(decoded_features);
        }
        else if (params->net_params.last_activation_function == "none")
        {
            decoded_features = decoded_features;
        }
        else if (params->net_params.last_activation_function == "leakyrelu")
        {
            decoded_features = torch::leaky_relu(decoded_features, 0.01);
        }
        else
        {
            CHECK(false);
        }

    }
#if 0
    else if (params->dataset_params.image_formation == "color_density")
    {
        // sigmoid for the color and relu for the density
        int num_channels = decoded_features.size(2);
        CHECK_GT(num_channels, 1);
        auto color       = decoded_features.slice(2, 0, num_channels - 1);
        auto density     = decoded_features.slice(2, num_channels - 1, num_channels);
        color            = torch::sigmoid(color);
        density          = torch::relu(density);
        decoded_features = torch::cat({color, density}, 2);
    }
    else
    {
        CHECK(false);
    }
#endif
    return decoded_features;
}

// void HierarchicalNeuralGeometry::AddParametersToOptimizer() {}
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::UniformSampledVolume(
//     std::vector<long> shape, int num_channels, vec3 roi_min, vec3 roi_max )
// {
//     Eigen::Vector<int, -1> shape_v;
//     shape_v.resize(shape.size());
//     for (int i = 0; i < shape.size(); ++i)
//     {
//         shape_v(i) = shape[i];
//     }

//     auto vol_shape                      = shape;
//     torch::Tensor output_volume_node_id = torch::zeros(vol_shape, torch::kLong);
//     torch::Tensor output_volume_valid   = torch::zeros(vol_shape, torch::kLong);

//     vol_shape.insert(vol_shape.begin(), num_channels);
//     torch::Tensor output_volume = torch::zeros(vol_shape);

//     PrintTensorInfo(output_volume);
//     // Generating the whole volume in one batch might cause a memory issue.
//     // Therefore, we batch the volume samples into a fixed size.
//     // tree->to(torch::kCPU);
//     auto all_samples = tree->UniformPhantomSamplesGPU(shape_v, false);
//     printf("run to sample gpu");
//     // all_samples.to(device);
//     tree->to(device);
//     int max_samples = 100000;
//     int num_batches = iDivUp(all_samples.size(), max_samples);
//     printf("could run to test here why\n");

//     ProgressBar bar(std::cout, "Sampling Volume", num_batches);

//     for (int i = 0; i < num_batches; ++i)
//     {
//         printf("wrting volume 1\n");
//         auto samples = all_samples.Slice(i * max_samples, std::min((i + 1) * max_samples, all_samples.size()));
//         samples.to(device);

//         auto b = tree->GroupSamplesPerNodeGPU(samples,  params->train_params.per_node_vol_batch_size);
//         b.to(device);
//         // auto model_output = pipeline->Forward(b);
//         // [num_groups, group_size, num_channels]
//         auto model_output = SampleVolumeBatched(b.global_coordinate, b.mask, b.node_ids);

//         model_output = model_output * b.integration_weight;


//         if (false)  //&&params->dataset_params.image_formation == "color_density"
//         {
//             model_output =
//                 model_output.slice(2, 0, num_channels) * model_output.slice(2, num_channels, num_channels + 1);
//         }

//         // [num_samples, num_channels]
//         auto cpu_out = model_output.cpu().contiguous().reshape({-1, num_channels});
//         // [num_samples]
//         auto cpu_mask   = b.mask.cpu().contiguous().reshape({-1});
//         auto cpu_weight = b.integration_weight.cpu().contiguous().reshape({-1});
//         // [num_samples]
//         auto cpu_ray_index = b.ray_index.cpu().contiguous().reshape({-1});

//         // [num_samples]
//         auto cpu_node_id = b.node_ids.cpu().contiguous().reshape({-1});



//         float* out_ptr       = cpu_out.template data_ptr<float>();
//         float* mask_ptr      = cpu_mask.template data_ptr<float>();
//         float* weight_ptr    = cpu_weight.template data_ptr<float>();
//         long* index_ptr      = cpu_ray_index.template data_ptr<long>();
//         long* node_index_ptr = cpu_node_id.template data_ptr<long>();

//         long num_samples = cpu_out.size(0);

//         for (int i = 0; i < num_samples; ++i)
//         {
//             if (mask_ptr[i] == 0) continue;
//             for (int c = 0; c < num_channels; ++c)
//             {
//                 float col = out_ptr[i * cpu_out.stride(0) + c * cpu_out.stride(1)];
//                 output_volume.data_ptr<float>()[output_volume.stride(0) * c + index_ptr[i]] = col;
//             }
//             auto out_node_id                                     = node_index_ptr[i / b.GroupSize()];
//             output_volume_node_id.data_ptr<long>()[index_ptr[i]] = out_node_id;
//             output_volume_valid.data_ptr<long>()[index_ptr[i]]   = weight_ptr[i];
//         }
//         bar.addProgress(1);
//     }
//     return {output_volume, output_volume_node_id, output_volume_valid};
// }

void HierarchicalNeuralGeometry::AddParametersToOptimizer() {}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry:: UniformSampledVolume(std::vector<long> shape, int num_channels,
                                                                                                        vec3 roi_min, vec3 roi_max , bool is_final)
{
    // torch::NoGradGuard ngg;

    if(is_final)
    {
        Eigen::Vector<int,  -1>   shape_v;
        Eigen::Vector<float, 3>  roi_min_v;
        Eigen::Vector<float, 3>  step_size_v;

        std::vector<long> shape_new;

        // std::vector<float> roi_min = params->octree_params.roi_min;
        // std::vector<float> roi_max = params->octree_params.roi_max;

        // Assume that input of the roi is correct but the input of the shape could be any
        // float step = std::min(std::min((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[1] - roi_min[1])/(shape[1]-1)), (roi_max[2] - roi_min[2])/(shape[2]-1));
        float step = std::max((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[2] - roi_min[2])/(shape[2]-1));

        // float step = std::max(std::max((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[1] - roi_min[1])/(shape[1]-1)), (roi_max[2] - roi_min[2])/(shape[2]-1));
        // float step = std::min(std::min((roi_max[0] - roi_min[0])/(shape[0]),(roi_max[1] - roi_min[1])/(shape[1])), (roi_max[2] - roi_min[2])/(shape[2]));

        for(int i = 0; i < shape.size(); ++i)
        {
            shape_new.push_back( int( std::round((roi_max[i] - roi_min[i])/step +1)) );
            // shape_new.push_back( int( std::round((roi_max[i] - roi_min[i])/step )) );

        }
        // printf("steps is %f %f\n", step, (roi_max[1] - roi_min[1])/step +1);
        shape_new[0] = shape[0];
        shape_new[2] = shape[2];

        shape_v.resize(shape_new.size());

        roi_min_v.resize(roi_min.size());
        step_size_v.resize(roi_min.size());
        for(int i = 0; i < shape_new.size(); ++i)
        {
            shape_v(i)      = shape_new[i];
            roi_min_v(i)    = roi_min[i];
            step_size_v(i)  = step;

        }

        std::cout << "final shape v " << shape_v(0) << " " << shape_v(1)<< " " << shape_v(2) << " " <<
                    roi_min_v(0) << " "<< roi_min_v(1)<< " " << roi_min_v(2) << " "
                    << step_size_v(0) << " "<< step_size_v(1)<< " "<< step_size_v(2) << " " <<
                    roi_max[0] << " "<< roi_max[1]<< " " << roi_max[2] << " " <<std::endl;


        auto vol_shape                      = shape_new;
        // torch::Tensor output_volume_node_id = torch::zeros(vol_shape, torch::kLong);
        // torch::Tensor output_volume_valid   = torch::zeros(vol_shape, torch::kLong);


        int volume_slice_num, slice_y_size;
        // CHECK_EQ(shape_new[1]%params->train_params.volume_slice_num, 0);
        bool needappend;
        if(shape_new[1]%params->train_params.volume_slice_num == 0)
        {
            slice_y_size = shape_new[1]/params->train_params.volume_slice_num;
            volume_slice_num = params->train_params.volume_slice_num;
            needappend = false;
        }
        else
        {
            slice_y_size = shape_new[1]/params->train_params.volume_slice_num +1 ;
            volume_slice_num = params->train_params.volume_slice_num  ;
            needappend = true;
        }

        vol_shape.insert(vol_shape.begin(), num_channels);
        int x_slice = shape_new[0];
        int z_slice = shape_new[2];
        // torch::Tensor output_volume = [];
        torch::Tensor output_volume;
        if(params->train_params.use_ground_truth_volume)
        {
            output_volume = torch::empty({x_slice, 0 ,z_slice});
        }
        else
        {
            output_volume = torch::empty({x_slice/2, 0, z_slice/2});
        }
        // torch::Tensor output_volume = torch::empty({shape_new[0], 0 ,shape_new[2]});
        torch::Tensor output_volume_node_id = torch::empty({shape_new[0], slice_y_size ,shape_new[2]});
        torch::Tensor output_volume_valid   = torch::empty({shape_new[0], slice_y_size ,shape_new[2]});
        // std::vector<float> roi_min_new;

        // for(int i = 0; i < roi_min.size(); i++)
        // {
        //     roi_min_new.push_back(roi_min[i]);
        // }

        // printf("volume_slice_num slice_y_size %d %d %f\n", volume_slice_num, slice_y_size, step);

        tree->to(device);
        int max_samples = 1048576;
        ProgressBar bar(std::cout, "Sampling Final Volume", volume_slice_num);

        for(int slice = 0 ; slice < volume_slice_num; slice++)
        {
            float slice_y = slice * slice_y_size * step + roi_min[1];
            // printf("slice value %f\n",(slice + 1) * slice_y_size * step);
            // if( (slice + 1) * slice_y_size * step > roi_max[1])

            if(roi_max[1] < slice_y)
            {
                printf("roi_max %f slice_y %f", roi_max[1], slice_y);
                continue;
            }
            if(slice == volume_slice_num-1 && needappend)
            {
                slice_y_size = int((roi_max[1] - slice_y)/step);
            }
            // if(slice_y_size <= 0)
            // {
            //     continue;
            // }
            torch::Tensor output_volume_slice = torch::zeros({shape_new[0], slice_y_size ,shape_new[2]});

            shape_v(1) = slice_y_size;
            roi_min_v(1) = slice_y;


            // auto all_samples = tree->UniformPhantomSamplesGPUbySlice( {shape_new[0], slice_y_size ,shape_new[2]}, false,
            // //                                                           {roi_min[0],slice_y,roi_min[2]}, {step, step, step});
            // printf("shape_v %d %d %d, step_size_v%f %f %f, roi_min_v %f %f %f roi_max %f slice_y %f\n", shape_v(0), shape_v(1), shape_v(2),
            //                                                                     step_size_v(0), step_size_v(1), step_size_v(2),
            //                                                                     roi_min_v(0), roi_min_v(1), roi_min_v(2), roi_max[1], slice_y);
            auto all_samples = tree->UniformPhantomSamplesGPUbySlice( shape_v, false, roi_min_v, step_size_v);


            // auto all_samples = tree->UniformPhantomSamplesGPU(shape_v, false);


            all_samples.to(device);
            int num_batches = iDivUp(all_samples.size(), max_samples);

            for(int i = 0; i < num_batches ; ++i)
            {
                auto samples = all_samples.Slice(i * max_samples, std::min((i + 1) * max_samples, all_samples.size()));
                auto b  = tree->GroupSamplesPerNodeGPU(samples,  params->train_params.per_node_vol_batch_size);
                b.to(device);
                // auto model_output = pipeline->Forward(b);
                // [num_groups, group_size, num_channels]

                auto model_output = SampleVolumeBatched(b.global_coordinate, b.mask, b.node_ids, params->net_params.using_decoder);


                model_output = model_output * b.integration_weight;
                if(false)
                {
                    model_output = model_output.slice(2, 0, num_channels) * model_output.slice(2, num_channels, num_channels + 1);
                }
                // [num_samples, num_channels]
                auto cpu_out = model_output.cpu().contiguous().reshape({-1, num_channels});
                // [num_samples]
                auto cpu_mask   = b.mask.cpu().contiguous().reshape({-1});
                // auto cpu_weight = b.integration_weight.cpu().contiguous().reshape({-1});
                // [num_samples]
                auto cpu_ray_index = b.ray_index.cpu().contiguous().reshape({-1});

                // [num_samples]
                // auto cpu_node_id = b.node_ids.cpu().contiguous().reshape({-1});



                float* out_ptr       = cpu_out.template data_ptr<float>();
                float* mask_ptr      = cpu_mask.template data_ptr<float>();
                // float* weight_ptr    = cpu_weight.template data_ptr<float>();
                long* index_ptr      = cpu_ray_index.template data_ptr<long>();
                // long* node_index_ptr = cpu_node_id.template data_ptr<long>();

                long num_samples = cpu_out.size(0);

                for (int j = 0; j < num_samples; ++j)
                {
                    if (mask_ptr[j] == 0) continue;
                    for (int c = 0; c < num_channels; ++c)
                    {
                        float col = out_ptr[j* cpu_out.stride(0) + c * cpu_out.stride(1)];
                        output_volume_slice.data_ptr<float>()[output_volume_slice.stride(0) * c + index_ptr[j]] = col;
                    }
                    // auto out_node_id                                     = node_index_ptr[j/ b.GroupSize()];
                    // output_volume_node_id.data_ptr<long>()[index_ptr[j]] = out_node_id;
                    // output_volume_valid.data_ptr<long>()[index_ptr[j]]   = weight_ptr[j];
                }
            }
            if(!params->train_params.use_ground_truth_volume)
            {
                output_volume = torch::cat({output_volume, output_volume_slice.slice(0,x_slice/4,3*x_slice/4).slice(2,z_slice/4, 3*z_slice/4)},1);
            }
            else
            {
                output_volume = torch::cat({output_volume, output_volume_slice},1);
            }
            // output_volume.append(output_volume_slice);
            // std::cout << "out put vol ";
            // PrintTensorInfo(output_volume);
            bar.addProgress(1);
        }

        // torch::Tensor output_volume = torch::stack(output_volume, 1);
        // PrintTensorInfo(output_volume);
        std::cout << "output volume " << TensorInfo(output_volume) << std::endl;

        return {output_volume.unsqueeze(0), output_volume_node_id, output_volume_valid};
    }
    else
    {
        Eigen::Vector<int, -1> shape_v;
        Eigen::Vector<float, 3> roi_min_v;
        Eigen::Vector<float, 3> step_size_v;


        float step = std::max((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[2] - roi_min[2])/(shape[2]-1));
        // float step = std::max(std::max((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[1] - roi_min[1])/(shape[1]-1)), (roi_max[2] - roi_min[2])/(shape[2]-1));
        // float step = std::min(std::min((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[1] - roi_min[1])/(shape[1]-1)), (roi_max[2] - roi_min[2])/(shape[2]-1));

        shape_v.resize(shape.size());
        for (int i = 0; i < shape.size(); ++i)
        {
            shape_v(i) = shape[i];
        }
        if(params->octree_params.use_quad_tree_rep)
        {
            shape_v(1) = int(std::round((roi_max[1] - roi_min[1])/step + 1));
        }
        for(int i = 0; i < shape_v.size(); ++i)
        {
            roi_min_v(i)    = roi_min[i];
            step_size_v(i)  = step;
        }

        auto vol_shape                      = shape;
        torch::Tensor output_volume_node_id = torch::zeros(vol_shape, torch::kLong);
        torch::Tensor output_volume_valid   = torch::zeros(vol_shape, torch::kLong);

        vol_shape.insert(vol_shape.begin(), num_channels);
        torch::Tensor output_volume = torch::zeros(vol_shape);

        // PrintTensorInfo(output_volume);
        tree->to(device);
        SampleList all_samples;
        if(params->octree_params.use_quad_tree_rep)
        {
            all_samples = tree->UniformPhantomSamplesGPUbySlice( shape_v, false, roi_min_v, step_size_v);
            // all_samples = tree->UniformPhantomSamplesGPU(shape_v, false);
        }
        else
        {
            all_samples = tree->UniformPhantomSamplesGPU(shape_v, false);
        }
        // auto all_samples = tree->UniformPhantomSamplesGPU(shape_v, false);
        // printf("run to sample gpu");
        // all_samples.to(device);

        // printf("tesht here 0\n");
        int max_samples = 1048576;
        int num_batches = iDivUp(all_samples.size(), max_samples);


        ProgressBar bar(std::cout, "Sampling Volume", num_batches);

        for (int i = 0; i < num_batches; ++i)
        {
            // printf("i is %d \n", i);
            // printf("wrting volume 1\n");
            auto samples = all_samples.Slice(i * max_samples, std::min((i + 1) * max_samples, all_samples.size()));
            samples.to(device);

            auto b = tree->GroupSamplesPerNodeGPU(samples,  params->train_params.per_node_vol_batch_size);
            // b.to(device);
            // // auto model_output = pipeline->Forward(b);
            // // [num_groups, group_size, num_channels]
            // printf("sample coordinate \n");
            // PrintTensorInfo(b.global_coordinate);

        auto local_samples = tree->ComputeLocalSamples(b.global_coordinate, b.node_ids);

        int x_slice = local_samples.size(0);
        int y_slice = local_samples.size(1);
        auto local_samples_slice = local_samples.slice(0, x_slice/4,3*x_slice/4).slice(1, x_slice/4, 3*y_slice/4);

        // PrintTensorInfo(local_samples_slice);
        // PrintTensorInfo(local_samples_slice.slice(2,0,1).abs());
        // PrintTensorInfo(local_samples_slice.slice(2,1,2).abs());
        // PrintTensorInfo(local_samples_slice.slice(2,2,3).abs());

            auto    model_output = SampleVolumeBatched(b.global_coordinate, b.mask, b.node_ids, params->net_params.using_decoder);

            // PrintTensorInfo(model_output);
            model_output = model_output * b.integration_weight;

            if (false)  //&&params->dataset_params.image_formation == "color_density"
            {
                model_output =
                    model_output.slice(2, 0, num_channels) * model_output.slice(2, num_channels, num_channels + 1);
            }

            // [num_samples, num_channels]
            auto cpu_out = model_output.cpu().contiguous().reshape({-1, num_channels});
            // [num_samples]
            auto cpu_mask   = b.mask.cpu().contiguous().reshape({-1});
            auto cpu_weight = b.integration_weight.cpu().contiguous().reshape({-1});
            // [num_samples]
            auto cpu_ray_index = b.ray_index.cpu().contiguous().reshape({-1});

            // [num_samples]
            auto cpu_node_id = b.node_ids.cpu().contiguous().reshape({-1});



            float* out_ptr       = cpu_out.template data_ptr<float>();
            float* mask_ptr      = cpu_mask.template data_ptr<float>();
            float* weight_ptr    = cpu_weight.template data_ptr<float>();
            long* index_ptr      = cpu_ray_index.template data_ptr<long>();
            long* node_index_ptr = cpu_node_id.template data_ptr<long>();

            long num_samples = cpu_out.size(0);


            // for (int ii = 0; ii < num_samples; ++ii)
            // int ii = num_samples - 1;
            int ii = 0;
            {
                if (mask_ptr[ii] == 0) continue;
                for (int c = 0; c < num_channels; ++c)
                {
                    float col = out_ptr[ii * cpu_out.stride(0) + c * cpu_out.stride(1)];
                    output_volume.data_ptr<float>()[output_volume.stride(0) * c + index_ptr[ii]] = col;
                }

                auto out_node_id                                     = node_index_ptr[ii / b.GroupSize()];

                output_volume_node_id.data_ptr<long>()[index_ptr[ii]] = out_node_id;

                output_volume_valid.data_ptr<long>()[index_ptr[ii]]   = weight_ptr[ii];

            }


            bar.addProgress(1);
        }
        return {output_volume, output_volume_node_id, output_volume_valid};
    }
}

void HierarchicalNeuralGeometry::SaveVolume(TensorBoardLogger* tblogger, std::string tb_name, std::string out_dir,
                                            int num_channels, float intensity_scale, int size, int slice_dim ,vec3 roi_min, vec3 roi_max )
{
    if (size <= 0) return;
    // std::cout << ">> Saving reconstructed volume." << std::endl;
    auto s = size;

    int mid_size = size;

    if(params->octree_params.use_quad_tree_rep)
    {
        mid_size = (int)s * (roi_max(1)- roi_min(1))/2;
    }
    auto [volume_density_raw, volume_node_id, volume_valid] = UniformSampledVolume({s, mid_size, s}, num_channels, roi_min, roi_max, false);

    std::cout << "Volume Raw: " << TensorInfo(volume_density_raw) << std::endl;
    // [channels, z, y, x]
    volume_density_raw = volume_density_raw * intensity_scale;

    // Normalize node error by volume
    torch::Tensor volume;
    if(params->octree_params.tree_depth ==0)
    {
        auto volumesize = tree->node_position_max - tree->node_position_min;
        volume = torch::prod(volumesize.cpu());
    }
    else
    {
        auto volumesize = tree->node_position_max - tree->node_position_min;
        volume = torch::prod(volumesize.cpu(), {1});
    } 

    // std::cout << "test here 0.4 " << TensorInfo(tree->node_max_density) << " " << TensorInfo(volume)<<std::endl;


    auto node_error = (tree->node_max_density.cpu().clamp_min(0) * tree->node_error.cpu().clamp_min(0) / volume).cpu();
    //        auto node_error = (tree->node_error / volume).cpu();

    auto error_volume = torch::index_select(node_error, 0, volume_node_id.reshape({-1}));
    auto max_density_volume =
        torch::index_select(tree->node_max_density.clamp_min(0).cpu(), 0, volume_node_id.reshape({-1}));

    error_volume       = error_volume.reshape({s, mid_size, s});
    max_density_volume = max_density_volume.reshape({1, s, mid_size, s});

    error_volume = error_volume / error_volume.max().clamp_min(0.01);
    // [3, s, s, s]
    error_volume = ColorizeTensor(error_volume, colorizePlasma);

    // set 0 nodes to full black
    error_volume = error_volume * (max_density_volume > 0);


    // draw culled nodes in different color
    error_volume                    = error_volume * volume_valid.to(torch::kFloat32);
    std::vector<float> culled_color = {0.5, 0., 0};
    torch::Tensor culled_color_t    = torch::from_blob(culled_color.data(), {3, 1, 1, 1});
    error_volume += (1 - volume_valid.to(torch::kFloat32)) * culled_color_t;



    // [s, 3, s , s]
    // error_volume = error_volume.permute({1, 0, 2, 3});

    // set node_id of inactive nodes to -1
    volume_node_id = (volume_node_id * volume_valid) - (1 - volume_valid);

    torch::Tensor volume_density = volume_density_raw;
    if (num_channels == 1)
    {
        // [z, y, x]
        volume_density = volume_density.squeeze(0);
        // [3, z, y, x]
        volume_density = ColorizeTensor(volume_density, colorizeMagma);
        // [z, 3, y, x]
        // volume_density = volume_density.permute({1, 0, 2, 3});
    }

    // skip channel
    slice_dim += 1;

    volume_node_id = volume_node_id.unsqueeze(0);

    // v = v.clamp(0, 1);
    // v = v / v.max();
    ProgressBar bar(std::cout, "Writing Volume", volume_density.size(slice_dim));
    for (int i = 0; i < volume_density.size(slice_dim); ++i)
    {
        CHECK_EQ(volume_node_id.dim(), 4);
        // [1, 1, h, w]
        auto node_index_img =
            volume_node_id.slice(slice_dim, i, i + 1).squeeze(slice_dim).unsqueeze(0).to(torch::kFloat);
        mat3 sobel_kernel;
        sobel_kernel << 0, 0, 0, -1, 0, 1, 0, 0, 0;
        // [1, h, w]
        auto structure_img_x = Filter2dIndependentChannels(node_index_img, sobel_kernel, 1).squeeze(0);
        auto structure_img_y = Filter2dIndependentChannels(node_index_img, sobel_kernel.transpose(), 1).squeeze(0);
        auto structure_img   = (structure_img_x.abs() + structure_img_y.abs()).clamp(0, 1);
        // [3, h, w]
        structure_img = structure_img.repeat({3, 1, 1});

        auto error_img   = error_volume.slice(slice_dim, i, i + 1).squeeze(slice_dim);
        auto density_img = volume_density.slice(slice_dim, i, i + 1).squeeze(slice_dim);

        structure_img = (error_img + structure_img).clamp(0, 1);


        auto density_structure = ImageBatchToImageRow(torch::stack({density_img, structure_img}));

        // LogImage(tblogger.get(), TensorToImage<ucvec3>(v[i]), "reconstruction_" +
        // std::to_string(epoch_id),
        //          i);
        //
        // LogImage(tblogger.get(), TensorToImage<ucvec3>(structure_img),
        //         "structure_" + std::to_string(epoch_id), i);

        auto saiga_img = TensorToImage<ucvec3>(density_structure);


        // std::filesystem::create_directories(out_dir);

        if (!out_dir.empty())
        {
            TensorToImage<ucvec3>(structure_img).save(out_dir + "/" + leadingZeroString(i, 5) + "_structure.png");
            TensorToImage<unsigned char>(volume_density_raw.slice(1, i, i + 1).squeeze(1))
                .save(out_dir + "/" + leadingZeroString(i, 5) + "_density.png");
        }
        LogImage(tblogger, saiga_img, tb_name, i);
        bar.addProgress(1);
    }
}
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::ComputeImage(SampleList all_samples, int num_channels, int num_pixels, bool use_decoder = true)
{
    torch::Tensor sample_output, weight, ray_index;

    torch::Tensor weight_bound;
    // try direct sampling. if it is not implemtend, an undefined tensor is returned
    sample_output = SampleVolumeIndirect(all_samples.global_coordinate, all_samples.node_id);


    torch::Tensor weight_bound_index_inv;
    if (sample_output.defined())
    {
        // if(weight_bound_index.defined())
        // {
        //     weight    = all_samples.weight.unsqueeze(1);
        //     ray_index = all_samples.ray_index;
        //     // weight_bound_index = torch::masked_select(weight_bound_index, weight_bound_index >= 0);
        //     // weight_bound = all_samples.weight.index_select(0, weight_bound_index);
        // }
        // else
        {
            weight    = all_samples.weight.unsqueeze(1);
            ray_index = all_samples.ray_index;
        }
    }
    else
    {
        // use indirect bachted sampling
        NodeBatchedSamples image_samples;

        {
            SAIGA_OPTIONAL_TIME_MEASURE("GroupSamplesPerNodeGPU", timer);
            // printf("input tensor info\n");
            // PrintTensorInfo(all_samples.global_coordinate);
            image_samples = tree->GroupSamplesPerNodeGPU(all_samples, params->train_params.per_node_batch_size);
            // PrintTensorInfo(image_samples.global_coordinate);
        }

        // [num_groups, group_size, num_channels] num_channels=1
        sample_output =
            SampleVolumeBatched(image_samples.global_coordinate, image_samples.mask, image_samples.node_ids, use_decoder);
        // printf("num channels %d\n", num_channels);


        weight    = image_samples.integration_weight;
        ray_index = image_samples.ray_index;

        // std::cout << "weight ray index " << TensorInfo(weight) << " " << TensorInfo(ray_index) << " " << TensorInfo(sample_output) << std::endl; 
    }

    // weight 763616 1
    // ray_index 763616
    // printf("ray information\n");
    // PrintTensorInfo(weight);
    // PrintTensorInfo(ray_index);

    torch::Tensor predicted_image, ratio, ratio_inv;
    {
        SAIGA_OPTIONAL_TIME_MEASURE("IntegrateSamplesXRay", timer);
        // [num_channels, num_rays]
        // printf("test here 0\n");
        // if(weight_bound_index.defined())
        // {
        //     std::tie(predicted_image,ratio, ratio_inv) = IntegrateSamplesXRay_Bound(sample_output, weight, ray_index, weight_bound_index , num_channels, num_pixels);
        // }
        // else
        {
            predicted_image = IntegrateSamplesXRay(sample_output, weight, ray_index, num_channels, num_pixels);
        }
        // printf("test here 1\n");
        CHECK_EQ(predicted_image.size(0), num_channels);
    }

    // if(predicted_image.requires_grad())
    // {
    //     TestVolume();
    // }

    // if(params->train_params.use_half_precision)
    // {
    //     predicted_image.to(torch::kFloat);
    // }
    // printf("predicted image\n");
    // // PrintTensorInfo(sample_output);
    // PrintTensorInfo(predicted_image);
    return {predicted_image, ratio, ratio_inv};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> HierarchicalNeuralGeometry::IntegrateSamplesXRay_Bound(torch::Tensor sample_values, torch::Tensor integration_weight, torch::Tensor ray_index,
                                                        torch::Tensor weight_bound_index, int num_channels, int num_rays)
{
    auto density_integral = torch::zeros({num_channels, num_rays}, torch::TensorOptions(ray_index.device()));
    torch::Tensor ratio, ratio_inv;
    if(sample_values.numel() == 0)
    {
        return {density_integral, ratio, ratio_inv};
    }
    CHECK_EQ(sample_values.dim(), integration_weight.dim());
    // auto weight_bound_index_gt = torch::masked_select(weight_bound_index, weight_bound_index >= 0);

    auto integration_weight_inroi = integration_weight.index_select(0, weight_bound_index);
    sample_values = sample_values.index_select(0, weight_bound_index);
    auto ray_index_inroi = ray_index.index_select(0, weight_bound_index);

    sample_values = sample_values * integration_weight_inroi;

    // [num_channels, num_samples]
    sample_values = sample_values.reshape({-1, num_channels}).permute({1,0});

    auto linear_ray_index = ray_index_inroi.reshape({-1});
    density_integral.index_add_(1, linear_ray_index, sample_values);


    auto weight_integral_in = torch::zeros_like(density_integral);
    auto weight_integral_all = torch::zeros_like(density_integral);
    weight_integral_all.index_add_(1, ray_index.reshape({-1}), integration_weight.reshape({-1, num_channels}).permute({1,0}));
    // PrintTensorInfo(linear_sample_output);
    // PrintTensorInfo(integration_weight);
    // PrintTensorInfo(integration_weight_inroi);
    // PrintTensorInfo(linear_ray_index);
    weight_integral_in.index_add_(1, linear_ray_index, integration_weight_inroi.reshape({-1, num_channels}).permute({1,0}));

    CHECK_EQ(density_integral.dim(),2);

    // printf("test here\n");
    // PrintTensorInfo(weight_integral_in);
    // PrintTensorInfo(weight_integral_all);

    std::tie(ratio, ratio_inv) = tree->ComputeRatio(weight_integral_in, weight_integral_all);
    // PrintTensorInfo(ratio);
    // if(density_integral.requires_grad())
    // {
    // printf("terst integral\n");
    // PrintTensorInfo(weight_integral_all);
    // // PrintTensorInfo(weight_integral_in);
    // // PrintTensorInfo(weight_integral_all - weight_integral_in);
    // PrintTensorInfo(weight_integral_in/weight_integral_all);
    // PrintTensorInfo(ratio);
    // PrintTensorInfo(ratio.min());
    // PrintTensorInfo(density_integral);
    // std::cout << weight_integral_all << std::endl;
    // // std::cout << density_integral << std::endl;
    // }


    return {density_integral, ratio, ratio_inv};
}


torch::Tensor NeuralGeometry::IntegrateSamplesXRay(torch::Tensor sample_values, torch::Tensor integration_weight,
                                                   torch::Tensor ray_index, int num_channels, int num_rays)
{
    // [num_channels, num_rays]
    auto density_integral = torch::zeros({num_channels, num_rays}, torch::TensorOptions(ray_index.device()));

    if (sample_values.numel() == 0)
    {
        return density_integral;
    }


    CHECK_EQ(sample_values.dim(), integration_weight.dim());
    sample_values = sample_values * integration_weight;

    // [num_channels, num_samples]
    auto linear_sample_output = sample_values.reshape({-1, num_channels}).permute({1, 0});

    // [num_samples]
    auto linear_ray_index = ray_index.reshape({-1});

    // std::cout << "linear index " << TensorInfo(linear_ray_index) << TensorInfo(linear_sample_output) << TensorInfo(density_integral) << std::endl;
    // PrintTensorInfo(density_integral);
    // PrintTensorInfo(linear_ray_index);
    // PrintTensorInfo(linear_sample_output);
    density_integral.index_add_(1, linear_ray_index, linear_sample_output);

    // if(density_integral.requires_grad())
    // {
    // auto weight_integral_all = torch::zeros_like(density_integral);
    // weight_integral_all.index_add_(1, ray_index.reshape({-1}), integration_weight.reshape({-1, num_channels}).permute({1,0}));
    // printf("test old integral\n");
    // PrintTensorInfo(weight_integral_all);
    // PrintTensorInfo(density_integral);
    // std::cout << weight_integral_all << std::endl;
    // std::cout << density_integral << std::endl;
    // }


    CHECK_EQ(density_integral.dim(), 2);

    return density_integral;
}


torch::Tensor NeuralGeometry::IntegrateSamplesAlphaBlending(torch::Tensor sample_values,
                                                            torch::Tensor integration_weight, torch::Tensor ray_index,
                                                            torch::Tensor sample_index_in_ray, int num_channels,
                                                            int num_rays, int max_samples_per_ray)
{
    // CHECK(params->dataset_params.image_formation == "color_density");
    if (sample_values.numel() == 0)
    {
        return torch::zeros({num_channels - 1, num_rays}, sample_values.device());
    }
    CHECK_GT(num_rays, 0);
    CHECK_GT(max_samples_per_ray, 0);
    CHECK_GT(num_channels, 0);
    CHECK(sample_index_in_ray.defined());

    //    std::cout << "Test IntegrateSamplesAlphaBlending " << num_rays << " " << max_samples_per_ray << std::endl;

    // Linearize all samples in one long array
    sample_values       = sample_values.reshape({-1, num_channels});
    ray_index           = ray_index.reshape({-1});
    sample_index_in_ray = sample_index_in_ray.reshape({-1});
    integration_weight  = integration_weight.reshape({-1, 1});

    // Multiply density by integration weight (the other channels are not modified)
    //    PrintTensorInfo(sample_values);
    //    PrintTensorInfo(integration_weight);
    sample_values.slice(1, num_channels - 1, num_channels) *= integration_weight;



    // [num_rays, max_samples_per_ray, num_channels]
    torch::Tensor sample_matrix  = torch::zeros({num_rays, max_samples_per_ray, num_channels},
                                                torch::TensorOptions(torch::kFloat).device(sample_values.device()));
    auto sample_offset_in_matrix = ray_index * max_samples_per_ray + sample_index_in_ray;
    auto tmp      = sample_matrix.reshape({-1, num_channels}).index_add_(0, sample_offset_in_matrix, sample_values);
    sample_matrix = tmp.reshape(sample_matrix.sizes());


    if (0)
    {
        // sum over the sample dimension
        // [num_channels, num_rays]
        auto simple_integral = sample_matrix.sum({1}).permute({1, 0});
        return simple_integral;
    }


    // We use the last channel as the density
    // [num_rays, num_samples_per_ray]
    auto density = sample_matrix.slice(2, num_channels - 1, num_channels).squeeze(2);
    //    PrintTensorInfo(density);

    // std::cout << density.slice(0,0,1).squeeze(0) << std::endl;

    // density to alpha
    // raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    // [num_rays, num_samples_per_ray]
    auto alpha = 1. - torch::exp(-density * 10);

    // std::cout << alpha[0] << std::endl;
    // PrintTensorInfo(alpha);
    // std::cout << alpha.slice(0,0,1).squeeze(0) << std::endl;

    // Exclusive prefix product on '1-alpha'
    //  weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    auto expanded_by_one     = torch::cat({torch::ones({num_rays, 1}, alpha.options()), 1. - alpha + 1e-10}, 1);
    auto prod                = torch::cumprod(expanded_by_one, 1);
    auto original_shape_prod = prod.slice(1, 0, max_samples_per_ray);
    CHECK_EQ(original_shape_prod.sizes(), alpha.sizes());
    // [num_rays, max_samples_per_ray]
    auto weights = alpha * original_shape_prod;
    // PrintTensorInfo(weights);
    // std::cout << density[0].slice(0, 0, 10) << std::endl;
    // std::cout << alpha[0].slice(0, 0, 10) << std::endl;
    // std::cout << weights[0].slice(0, 0, 10) << std::endl;
    // std::cout << original_shape_prod[0] << std::endl;
    // exit(0);

    // The remaining values are the color
    // [num_rays, max_samples_per_ray, num_color_channels]
    auto rgb = sample_matrix.slice(2, 0, num_channels - 1);

    //    std::cout << "density" << std::endl;
    //    PrintTensorInfo(density);
    //
    //    std::cout << "rgb" << std::endl;
    //    PrintTensorInfo(rgb.slice(2, 0, 1));
    //    PrintTensorInfo(rgb.slice(2, 1, 2));
    //    PrintTensorInfo(rgb.slice(2, 2, 3));
    //    PrintTensorInfo(rgb);

    // weighted sum over density values
    // [num_rays, num_color_channels]
    auto rgb_map = torch::sum(weights.unsqueeze(2) * rgb, {1});
    //    PrintTensorInfo(rgb_map);



    return rgb_map.permute({1, 0});
}

// void HierarchicalNeuralGeometry::TestVolume()
// {
//     torch::NoGradGuard ngg;
//     vec3 roi_min = {-1.0,-0.375,-1.0};
//     vec3 roi_max = {1.0,0,1.0};
//     std::vector<long> shape = {1024,1024,1024};
//     float step = std::max(std::max((roi_max[0] - roi_min[0])/(shape[0] -1),(roi_max[1] - roi_min[1])/(shape[1]-1)), (roi_max[2] - roi_min[2])/(shape[2]-1));
//     std::vector<long> shape_new;
//     for(int i = 0; i < shape.size(); ++i)
//     shape_new.push_back( int( std::round((roi_max[i] - roi_min[i])/step +1)) );
//     shape_new[0] = shape[0];
//     shape_new[2] = shape[2];

//     Eigen::Vector<int, -1> shape_v;
//     Eigen::Vector<float, 3> roi_min_v;
//     Eigen::Vector<float, 3> step_size_v;
//     shape_v.resize(shape_new.size());
//     roi_min_v.resize(roi_min.size());
//     step_size_v.resize(roi_min.size());
//     for(int i = 0; i < shape_new.size(); ++i)
//     {
//         shape_v(i)      = shape_new[i];
//         roi_min_v(i)    = roi_min[i];
//         step_size_v(i)  = step;
//     }
//     int slice = 112;
//     int slice_y_size = 1;
//     float slice_y = slice * slice_y_size * step + roi_min[1];
//             // tree->to(device);
//     roi_min_v(1) = slice_y;
//     shape_v(1) = 1;
//     int max_samples = 1048576;
//     auto all_samples = tree->UniformPhantomSamplesGPUbySlice( shape_v, false, roi_min_v, step_size_v);
//     all_samples.to(device);
//     int num_batches = iDivUp(all_samples.size(), max_samples);
//     torch::Tensor output_volume_slice = torch::zeros({shape_new[0], slice_y_size ,shape_new[2]});
//     std::string fold_dir = "/home/wangy0k/Desktop/owntree/hyperacornquad/Experiments/local_coord/";
//         std::cout << "test shape v " << shape_v(0) << shape_v(1) << shape_v(2) << " " <<
//                     roi_min_v(0) << roi_min_v(1) << roi_min_v(2) << " " << step_size_v(0) << step_size_v(1)<< step_size_v(2) << std::endl;

//     for(int i = 0; i < num_batches ; ++i)
//     {
//         auto samples = all_samples.Slice(i * max_samples, std::min((i + 1) * max_samples, all_samples.size()));
//         auto b  = tree->GroupSamplesPerNodeGPU(samples,  params->train_params.per_node_vol_batch_size);
//         b.to(device);
//         // auto model_output = pipeline->Forward(b);
//         // [num_groups, group_size, num_channels]
//         auto model_output = SampleVolumeBatched(b.global_coordinate, b.mask, b.node_ids, params->net_params.using_decoder);
//         // PrintTensorInfo(model_output);



//         auto local_samples = tree->ComputeLocalSamples(b.global_coordinate, b.node_ids);

//         int x_slice = local_samples.size(0);
//         int y_slice = local_samples.size(1);
//         auto local_samples_slice = local_samples.slice(0, x_slice/4,3*x_slice/4).slice(1, x_slice/4, 3*y_slice/4);
//         for(int ii = 0; ii < 3; ++ii)
//         {
//             auto im1 = TensorToImage<float>(local_samples_slice.slice(2,ii,ii+1).squeeze(2).unsqueeze(0).unsqueeze(0));
//             auto colorized = ImageTransformation::ColorizeTurbo(im1);
//             TemplatedImage<unsigned short> im1_new(im1.dimensions());
//             for(int iix : im1.rowRange())
//             {
//                 for(int iiy : im1.colRange())
//                 {
//                     im1_new(iix,iiy) = im1(iix,iiy) * std::numeric_limits<unsigned short>::max();
//                 }
//             }
//             im1_new.save(fold_dir +std::to_string(ii)+"local_samples.png");
//         }

//         // PrintTensorInfo(local_samples.slice(2,0,1).abs());
//         // PrintTensorInfo(local_samples.slice(2,2,3).abs());

//         model_output = model_output * b.integration_weight;
//         if(false)
//         {
//             model_output = model_output.slice(2, 0, num_channels) * model_output.slice(2, num_channels, num_channels + 1);
//         }
//         float factor = std::log(1.1326878071/0.2232130319);
//         model_output = torch::exp(-factor * model_output);
//         // [num_samples, num_channels]
//         auto cpu_out = model_output.cpu().contiguous().reshape({-1, num_channels});
//         // [num_samples]
//         auto cpu_mask   = b.mask.cpu().contiguous().reshape({-1});
//         // auto cpu_weight = b.integration_weight.cpu().contiguous().reshape({-1});
//         // [num_samples]
//         auto cpu_ray_index = b.ray_index.cpu().contiguous().reshape({-1});

//         // [num_samples]
//         // auto cpu_node_id = b.node_ids.cpu().contiguous().reshape({-1});

//         float* out_ptr       = cpu_out.template data_ptr<float>();
//         float* mask_ptr      = cpu_mask.template data_ptr<float>();
//         long* index_ptr      = cpu_ray_index.template data_ptr<long>();

//         long num_samples = cpu_out.size(0);

//         for (int j = 0; j < num_samples; ++j)
//         {
//             if (mask_ptr[j] == 0) continue;
//             for (int c = 0; c < num_channels; ++c)
//             {
//                 float col = out_ptr[j* cpu_out.stride(0) + c * cpu_out.stride(1)];
//                 output_volume_slice.data_ptr<float>()[output_volume_slice.stride(0) * c + index_ptr[j]] = col;
//             }
//         }
//         auto im1 = TensorToImage<float>(output_volume_slice.squeeze(1).slice(0, x_slice/4,3*x_slice/4).slice(1, x_slice/4, 3*y_slice/4).unsqueeze(0).unsqueeze(0));
//         auto colorized = ImageTransformation::ColorizeTurbo(im1);
//         TemplatedImage <unsigned short> im1_new(im1.dimensions());
//         for(int iix : im1.rowRange())
//         {
//             for(int iiy : im1.colRange())
//             {
//                 im1_new(iix,iiy) = im1(iix,iiy) * std::numeric_limits<unsigned short>::max();
//             }
//         }
//         im1_new.save(fold_dir +"density.png");
//     }
//     // printf("output volume \n");
//     // PrintTensorInfo(output_volume_slice);.slice(0, x_slice/4,3*x_slice/4).slice(1, x_slice/4, 3*y_slice/4);

//     // char c = getchar();
// }
