#include "geometry_tensor_qo.h"
#include "modules/IndirectGridSample3D.h"

#include "saiga/vision/torch/ColorizeTensor.h"
#define epsilon 0.0001
GeometryTensorQO::GeometryTensorQO(int num_channels, int D, HyperTreeBase tree, std::shared_ptr<CombinedParams> params)
    :HierarchicalNeuralGeometry(num_channels, D, params, tree)
{
    std::cout <<"Try Quad Oct Methods " << std::endl;
    std::vector<long> feature_grid_shape, feature_grid_shape_vec;
    auto roi_min = params->octree_params.tree_optimizer_params.tree_roi_min;
    auto roi_max = params->octree_params.tree_optimizer_params.tree_roi_max;
    for(int i = 0; i < D; ++i)
    {
        vec_grid_size[i] = (int) params->net_params.grid_size;
        if(params->net_params.use_tree_line)
        {
            feature_grid_shape_vec.push_back((int) params->net_params.grid_size);
        }
        else
        {
            feature_grid_shape_vec.push_back((int)std::round((params->net_params.grid_size -1) * std::pow(2, params->octree_params.tree_depth) * (params->octree_params.tree_optimizer_params.tree_roi_max(i) - params->octree_params.tree_optimizer_params.tree_roi_min(i))/2.));
        }
    }
    vec_grid_size[1] = (int) std::round(params->net_params.grid_size * std::pow(2, params->octree_params.tree_depth) * (roi_max(1)-roi_min(1)/2));
    feature_grid_shape_vec[1] = vec_grid_size[1];
    feature_grid_shape_vec[2] = (int)std::round((params->net_params.grid_size -1) * std::pow(2, params->octree_params.tree_depth));
    line_tv_index_z[0] = (params->octree_params.tree_optimizer_params.tree_roi_min(2)+1)/2.0f * feature_grid_shape_vec[2];
    line_tv_index_z[1] = (params->octree_params.tree_optimizer_params.tree_roi_max(2)+1)/2.0f * feature_grid_shape_vec[2];

    std::cout <<"vec grid size " << vec_grid_size[0] << " " << vec_grid_size[1] << " " << vec_grid_size[2] << std::endl;
    // std::cout <<"line tv index " << line_tv_index[0] <<" " << line_tv_index[1] << std::endl;
    for(int i = 0; i < D; ++i)
    {
        feature_grid_shape.push_back(vec_grid_size[i]);
    }
    int grid_feature = params->net_params.grid_features;
    printf("grid_feature %d \n", grid_feature);
    std::cout << grid_feature %3 << std::endl;

    std::vector<int> density_n_comp;
    auto vec_feature = params->net_params.vec_grid_feature;
    for(int i = 0; i < D; ++i)
    {
        if(params->train_params.use_vec_feature)
        {
            density_n_comp.push_back(vec_feature(i));
        }
        else
        {
            density_n_comp.push_back(grid_feature);
        }
    }

    torch_plane_index = torch::tensor({{1,2},{0,2},{0,1}}, torch::TensorOptions(torch::kInt32).device(device));
    // int feature_in      = 3 * grid_feature;
    // int feature_in = vec_feature(0) + vec_feature(1) + vec_feature(2);
    int feature_in = density_n_comp[0] + density_n_comp[1] + density_n_comp[2];
    if((! (params->train_params.plane_op == "cat")))
    {
        feature_in = density_n_comp[0];
    }
    if(params->train_params.tensor_op == "cat")
    {
        feature_in = density_n_comp[0] + density_n_comp[1] + density_n_comp[2];
    }
    std::cout << "input feature " << feature_in << std::endl;
    std::cout << "density n comp is " << density_n_comp << std::endl;
    std::cout << "use ground truth volume " <<  params->train_params.use_ground_truth_volume << std::endl;
    int decoder_num_channels = num_channels;

    if(!params->net_params.using_decoder)
    {
        decoder = FCBlock(feature_in, decoder_num_channels, params->net_params.decoder_hidden_layers,
                params->net_params.decoder_hidden_features, true, params->net_params.decoder_activation);
        register_module("decoder", decoder);

        optimizer_decoder =
            std::make_shared<torch::optim::Adam>(std::vector<torch::Tensor>(), torch::optim::AdamOptions().lr(10));
        std::cout << "Optimizing Decoder with LR " << params->net_params.decoder_lr << std::endl;
        auto options = std::make_unique<torch::optim::AdamOptions>(params->net_params.decoder_lr);
        optimizer_decoder->add_param_group({decoder->parameters(), std::move(options)});
    }
    else
    {
        std::cerr << "wrongly define the decoder using in quad tensor " << std::endl;
    }

    int num_nodes_inroi = std::pow(2, params->octree_params.tree_depth-1) * std::pow(2, params->octree_params.tree_depth-1);
    if(params->train_params.use_ground_truth_volume)
    {
        num_nodes_inroi = params->octree_params.tree_optimizer_params.max_active_nodes_initial;
    }

    tensor_plane_vector = Tensor_PlaneQuad(num_nodes_inroi,
                                            matMode, density_n_comp, feature_grid_shape, params->train_params.vm_init_scale,
                                            params->train_params.grid_init);
    if(!params->train_params.plane_vec_only)
    {
        std::vector<long> feature_grid_shape_line_vec;
        for(int i = 0; i < D; ++i)
        {
            feature_grid_shape_line_vec.push_back(feature_grid_shape_vec[i]);
        }
        int num_nodes = params->octree_params.tree_optimizer_params.max_active_nodes_initial;
        if(! params->net_params.use_tree_line)
        {
            num_nodes = 1;
        }
        tensor_line_vector  = Tensor_LineQuad(num_nodes,
                                            vecMode, density_n_comp, feature_grid_shape_line_vec, params->train_params.vm_init_scale,
                                            params->train_params.grid_init);
    }

    // // add octant
    // grid_sampler = NeuralGridSampler(false, params->net_params.sampler_align_corners);
    // register_module("grid_sampler", grid_sampler);
    std::vector<long> feature_grid_shape_3d;
    for(int i = 0; i < D; ++i)
    {
        feature_grid_shape_3d.push_back(feature_grid_shape[i] * params->train_params.out_roi_scale);
    }

    int num_nodes_outroi_tree = std::pow(2, params->octree_params.start_layer-1) * std::pow(2, params->octree_params.start_layer-1);
    if(!params->train_params.use_ground_truth_volume)
    {
        int num_nodes_outroi = std::max(num_nodes_outroi_tree,params->octree_params.tree_optimizer_params.max_active_nodes_initial - num_nodes_inroi );
        explicit_grid_generator =
                ExplicitFeatureGrid(num_nodes_outroi,
                                    feature_in, feature_grid_shape_3d, params->train_params.grid_init, params->train_params.vm_init_scale);
        register_module("explicit_grid_generator", explicit_grid_generator);
    }



    register_module("tensor_plane_vector", tensor_plane_vector);
    if(!params->train_params.plane_vec_only)
    {
        register_module("tensor_line_vector", tensor_line_vector);
    }

    std::cout << "tensor plane vecor "  << TensorInfo(tensor_plane_vector->plane_coef_yz) << std::endl
                                        << TensorInfo(tensor_plane_vector->plane_coef_xz) << std::endl
                                        << TensorInfo(tensor_plane_vector->plane_coef_xy) << std::endl;
    if(!params->train_params.plane_vec_only)
    {
        std::cout << "tensor line vecor "  << TensorInfo(tensor_line_vector->line_coef_x) << std::endl
                                            << TensorInfo(tensor_line_vector->line_coef_y) << std::endl
                                            << TensorInfo(tensor_line_vector->line_coef_z) << std::endl;
    }
    auto memory_size = tensor_plane_vector->plane_coef_yz.numel() + tensor_plane_vector->plane_coef_xz.numel() + tensor_plane_vector->plane_coef_xy.numel();
    if(!params->train_params.plane_vec_only)
    {
        memory_size += tensor_plane_vector->plane_coef_yz.numel() + tensor_plane_vector->plane_coef_xz.numel() + tensor_plane_vector->plane_coef_xy.numel()+
                        tensor_line_vector->line_coef_x.numel() + tensor_line_vector->line_coef_y.numel() + tensor_line_vector->line_coef_z.numel();

    }
    if(!params->train_params.use_ground_truth_volume)
    {
        memory_size += explicit_grid_generator->grid_data.numel();
    }

    std::cout <<"Numel " << memory_size
              <<" Memory " << (memory_size) * sizeof(float)/ 1000000.0 <<" MB" <<std::endl;
    int num = 0;
    for(auto p : tensor_plane_vector->parameters())
    {
        num += p.numel();
    }
    for(auto p : tensor_line_vector->parameters())
    {
        num += p.numel();
    }
    if(!params->train_params.use_ground_truth_volume)
    {
    for(auto p : explicit_grid_generator->parameters())
    {
        num += p.numel();
    }
    }
    std::cout << "params is " << num/ 1000000.0 << std::endl;
    // std::vector<torch::Tensor> tensor_plane_bak = tensor_plane_vector->parameters();
    // PrintTensorInfo(sample_output);
    // Loss perform
    {
        if(params->train_params.loss_tv > 0)
        {
            printf("perform TV loss \n");
        }
        if(params->train_params.loss_edge > 0)
        {
            printf("perform EDGE loss \n");
        }

    }
    std::cout << "=== ============= ===" << ConsoleColor::RESET << std::endl;
}

void GeometryTensorQO::AddParametersToOptimizer()
{
    HierarchicalNeuralGeometry::AddParametersToOptimizer();
    if(params->train_params.exex_op == "adam")
    {
        std::cout << "Optimizing Quad Tensor Radiance field with (ADAM) LR " << params->train_params.lr_exex_grid_adam << std::endl;
        printf("add paramters \n");
        PrintTensorInfo(tensor_plane_vector->plane_coef_yz);
        optimizer_adam->add_param_group(
            {tensor_plane_vector->parameters(),
            std::make_unique<torch::optim::AdamOptions>(params->train_params.lr_exex_grid_adam)});
        if(!params->train_params.plane_vec_only)
        {
            optimizer_adam->add_param_group(
                {tensor_line_vector->parameters(),
                std::make_unique<torch::optim::AdamOptions>(params->train_params.lr_exex_grid_adam)});
        }
        if(!params->train_params.use_ground_truth_volume)
        {
            optimizer_adam->add_param_group(
                    {explicit_grid_generator->parameters(),
                        std::make_unique<torch::optim::AdamOptions>(params->train_params.lr_exex_grid_adam )});

        }

    }
    else if (params->train_params.exex_op == "rms")
    {
        std::cout << "Optimizing Quad Tensor Radiance with (RMS) LR " << params->train_params.lr_exex_grid_rms
                    << std::endl;
        printf("add parameters \n");
        PrintTensorInfo(tensor_plane_vector->plane_coef_yz);
        // // error happens here to debug
        optimizer_rms->add_param_group(
            {tensor_plane_vector->parameters(),
            std::make_unique<torch::optim::RMSpropOptions>(params->train_params.lr_exex_grid_rms)});
        if(!params->train_params.plane_vec_only)
        {
            optimizer_rms->add_param_group(
                {tensor_line_vector->parameters(),
                std::make_unique<torch::optim::RMSpropOptions>(params->train_params.lr_exex_grid_rms)});
        }
        if(!params->train_params.use_ground_truth_volume)
        {
            optimizer_rms->add_param_group(
                    {explicit_grid_generator->parameters(),
                    std::make_unique<torch::optim::RMSpropOptions>(params->train_params.lr_exex_grid_rms )});

        }

    }
    else
    {
        CHECK(false);
    }

}

torch::Tensor GeometryTensorQO::SampleVolumeIndirect_feature(torch::Tensor global_coordinate, torch::Tensor node_id)
{
    // torch::Tensor local_samples, neural_features;
    torch::Tensor grid_plane_yz, grid_plane_xz, grid_plane_xy, grid_line_x, grid_line_y, grid_line_z, grid;
    if(global_coordinate.numel() == 0)
    {
        return global_coordinate;
    }
    torch::Tensor node_id_inroi, node_id_outroi, global_coordinate_inroi, global_coordinate_outroi;
    torch::Tensor node_id_in_roi_index, node_id_out_roi_index;
    {
        torch::Tensor node_id_index = torch::arange({node_id.size(0)}, torch::TensorOptions(node_id.device()).dtype(torch::kLong));
        auto node_id_in_roi_id = tree->ComputeBoundWeight(node_id, in_roi_node_index, false);
        node_id_in_roi_index = torch::masked_select(node_id_index, node_id_in_roi_id >= 0);
        node_id_out_roi_index = torch::masked_select(node_id_index, node_id_in_roi_id < 0);
        node_id_inroi = torch::index_select(node_id, 0, node_id_in_roi_index);
        node_id_outroi = torch::index_select(node_id, 0, node_id_out_roi_index);

        global_coordinate_inroi = torch::index_select(global_coordinate, 0, node_id_in_roi_index);
        global_coordinate_outroi = torch::index_select(global_coordinate, 0, node_id_out_roi_index);
    }
    torch::Tensor neural_features_inroi;
    {
        auto local_samples = tree->ComputeLocalSamples(global_coordinate_inroi, node_id_inroi);
        auto local_node_id = torch::index_select(node_active_prefix_sum_inroi, 0, node_id_inroi.reshape({-1}).contiguous());
        auto line_x_coord = local_samples.slice(1,0,1);
        auto line_y_coord = local_samples.slice(1,1,2);
        auto line_z_coord = local_samples.slice(1,2,3);

        auto plane_yz = torch::cat({line_y_coord, line_z_coord},1);
        auto plane_xz = torch::cat({line_x_coord, line_z_coord},1);
        auto plane_xy = torch::cat({line_x_coord, line_y_coord},1);
        int num_samples = local_samples.sizes()[0];
        if(!params->train_params.plane_vec_only)
        {
            torch::Tensor vec_node_id = torch::zeros(num_samples, torch::TensorOptions(torch::kLong).device(local_samples.device()));
            auto global_coordinate_tmp = global_coordinate_inroi.reshape({-1,3});
            line_x_coord = 2. * (global_coordinate_tmp.slice(1,0,1)-params->octree_params.tree_optimizer_params.tree_roi_min(0))/(params->octree_params.tree_optimizer_params.tree_roi_max(0) - params->octree_params.tree_optimizer_params.tree_roi_min(0)) -1;
                // line_y_coord = global_coordinate_tmp.slice(1,1,2);
            // line_z_coord = 2. * (global_coordinate_tmp.slice(1,2,3)-params->octree_params.tree_optimizer_params.tree_roi_min(2))/(params->octree_params.tree_optimizer_params.tree_roi_max(2) - params->octree_params.tree_optimizer_params.tree_roi_min(2)) -1;
            line_z_coord = global_coordinate_tmp.slice(1,2,3);
            // printf("test line coord\n");
            // PrintTensorInfo(line_x_coord);
            // PrintTensorInfo(line_z_coord);

            if(params->train_params.tensor_op == "cat")
            {
                auto mat_feat = torch::empty({num_samples, 0},
                                                        torch::TensorOptions(torch::kFloat).device(local_samples.device()));
                mat_feat = torch::cat({mat_feat, IndirectGridSample2D(tensor_plane_vector->forward( 0), local_node_id, plane_yz)}, 1);
                mat_feat = torch::cat({mat_feat, IndirectGridSample2D(tensor_plane_vector->forward( 1), local_node_id, plane_xz)}, 1);
                mat_feat = torch::cat({mat_feat, IndirectGridSample2D(tensor_plane_vector->forward( 2), local_node_id, plane_xy)}, 1);
                auto vec_feat = torch::empty({num_samples, 0},
                                                        torch::TensorOptions(torch::kFloat).device(local_samples.device()));
                vec_feat = torch::cat({vec_feat, IndirectGridSample1D(tensor_line_vector->forward(0), vec_node_id, line_x_coord)}, 1);
                vec_feat = torch::cat({vec_feat, IndirectGridSample1D(tensor_line_vector->forward(1), vec_node_id, line_y_coord)}, 1);
                vec_feat = torch::cat({vec_feat, IndirectGridSample1D(tensor_line_vector->forward(2), vec_node_id, line_z_coord)}, 1);
                neural_features_inroi = mat_feat * vec_feat;
            }
            else if(params->train_params.tensor_op == "add")
            {
                auto mat_feat = IndirectGridSample2D(tensor_plane_vector->forward( 0), local_node_id, plane_yz);
                auto vec_feat = IndirectGridSample1D(tensor_line_vector->forward(0), vec_node_id, line_x_coord);
                neural_features_inroi = mat_feat * vec_feat;
                mat_feat = IndirectGridSample2D(tensor_plane_vector->forward( 1), local_node_id, plane_xz);
                vec_feat = IndirectGridSample1D(tensor_line_vector->forward(1), vec_node_id, line_y_coord);
                neural_features_inroi += mat_feat * vec_feat;
                mat_feat = IndirectGridSample2D(tensor_plane_vector->forward( 2), local_node_id, plane_xy);
                vec_feat = IndirectGridSample1D(tensor_line_vector->forward(2), vec_node_id, line_z_coord);
                neural_features_inroi += mat_feat * vec_feat;
            }
            else
            {
                throw std::runtime_error{"current tenosr support operation cat add"};
            }

        }
        // std::cout << "in roi node " <<std::get<0>(at::_unique(node_id_inroi)) << std::endl;
        // PrintTensorInfo(local_samples);
        // PrintTensorInfo(local_node_id);
    }
    // PrintTensorInfo(neural_features_inroi);
    torch::Tensor neural_features_outroi;
    {
        if(!params->train_params.use_ground_truth_volume)
        {
            auto local_samples = tree->ComputeLocalSamples(global_coordinate_outroi, node_id_outroi);
            auto local_node_id = torch::index_select(node_active_prefix_sum_outroi, 0, node_id_outroi.reshape({-1}).contiguous());
            // PrintTensorInfo(local_samples);
            // PrintTensorInfo(local_node_id);
            neural_features_outroi = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
            // auto neural_featurestest = grid_sampler->forward(explicit_grid_generator->forward(local_node_id), local_samples);

            // PrintTensorInfo(neural_features_outroi);
            // PrintTensorInfo(neural_features_outroi- neural_featurestest);
            // char c = getchar();
        }

    }
    torch::Tensor neural_features = torch::zeros({global_coordinate.size(0),neural_features_inroi.size(1)}, torch::TensorOptions(neural_features_inroi.dtype()).device(global_coordinate.device()));

    // printf("teste neural features");
    // PrintTensorInfo(neural_features);
    // PrintTensorInfo(neural_features_inroi);
    // PrintTensorInfo(neural_features_outroi);
    neural_features.index_add_(0, node_id_in_roi_index,neural_features_inroi);
    if(!params->train_params.use_ground_truth_volume)
    {
        neural_features.index_add_(0, node_id_out_roi_index,neural_features_outroi);
    }
    // printf("test here\n");
    // torch::Tensor nodeind = torch::zeros({global_coordinate.size(0),1}, torch::TensorOptions(node_id_in_roi_index.dtype()).device(node_id_in_roi_index.device()));
    // nodeind.index_add_(0,node_id_in_roi_index,node_id_in_roi_index.unsqueeze(1) );
    // nodeind.index_add_(0,node_id_out_roi_index,node_id_out_roi_index.unsqueeze(1) );
    //     torch::Tensor node_id_index = torch::arange({node_id.size(0)}, torch::TensorOptions(node_id.device()).dtype(torch::kLong));
    // printf("test last");
    // PrintTensorInfo(nodeind);
    // PrintTensorInfo( node_id_index);
    // PrintTensorInfo(neural_features);
    // PrintTensorInfo(explicit_grid_generator->grid_data);


    return neural_features;
}
torch::Tensor GeometryTensorQO::SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id)
{
    if(global_coordinate.numel() == 0)
    {
        return global_coordinate;
    }
    auto feature = SampleVolumeIndirect_feature(global_coordinate, node_id);
    {
        SAIGA_OPTIONAL_TIME_MEASURE("DecodeFeatures", timer);
        feature = DecodeFeatures(feature);
        feature = feature.contiguous();
        // PrintTensorInfo(density);
    }
    return feature;

}
torch::Tensor GeometryTensorQO::SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id)
{
    // auto local_samples = tree->ComputeLocalSamples(global_coordinate, node_id);
    // printf("test here the sample shapes \n");
    // PrintTensorInfo(local_samples);
    // PrintTensorInfo(global_coordinate);
    // PrintTensorInfo(node_id);
    // auto local_node_id = tree->GlobalNodeIdToLocalActiveId(node_id);
    // PrintTensorInfo(local_node_id);
    if (global_coordinate.numel() == 0)
    {
        // No samples -> just return and empty tensor
        return global_coordinate;
    }
    torch::Tensor neural_features;
    {
        auto node_id_span = node_id.unsqueeze(1).expand({-1,global_coordinate.size(1)}).reshape({-1});
        auto global_coordinate_span = global_coordinate.reshape({-1,3});

        neural_features = SampleVolumeIndirect_feature(global_coordinate_span, node_id_span);
    }
    // printf("test feature\n");
    // PrintTensorInfo(neural_features);

    neural_features = neural_features.reshape({global_coordinate.size(0), global_coordinate.size(1), -1});
    // PrintTensorInfo(neural_features);
    return neural_features;
}

std::tuple<torch::Tensor, torch::Tensor> GeometryTensorQO::SampleVolumeIndirect_edge(torch::Tensor global_coordinate, torch::Tensor node_id)
{
        // torch::Tensor local_samples, neural_features;
    torch::Tensor grid_plane_yz, grid_plane_xz, grid_plane_xy, grid_line_x, grid_line_y, grid_line_z, grid;
    if(global_coordinate.numel() == 0)
    {
        return {global_coordinate, torch::Tensor()};
    }
    torch::Tensor node_id_inroi, node_id_outroi, global_coordinate_inroi, global_coordinate_outroi;
    torch::Tensor node_id_in_roi_index, node_id_out_roi_index;
    {
        torch::Tensor node_id_index = torch::arange({node_id.size(0)}, torch::TensorOptions(node_id.device()).dtype(torch::kLong));
        auto node_id_in_roi_id = tree->ComputeBoundWeight(node_id, in_roi_node_index, false);
        node_id_in_roi_index = torch::masked_select(node_id_index, node_id_in_roi_id >= 0);
        node_id_out_roi_index = torch::masked_select(node_id_index, node_id_in_roi_id < 0);
        node_id_inroi = torch::index_select(node_id, 0, node_id_in_roi_index);
        node_id_outroi = torch::index_select(node_id, 0, node_id_out_roi_index);

        global_coordinate_inroi = torch::index_select(global_coordinate, 0, node_id_in_roi_index);
        global_coordinate_outroi = torch::index_select(global_coordinate, 0, node_id_out_roi_index);
    }
    torch::Tensor neural_features_inroi;
    {
        auto local_samples = tree->ComputeLocalSamples(global_coordinate_inroi, node_id_inroi);
        auto local_node_id = torch::index_select(node_active_prefix_sum_inroi, 0, node_id_inroi.reshape({-1}).contiguous());
        auto line_x_coord = local_samples.slice(1,0,1);
        auto line_y_coord = local_samples.slice(1,1,2);
        auto line_z_coord = local_samples.slice(1,2,3);

        auto plane_yz = torch::cat({line_y_coord, line_z_coord},1);
        auto plane_xz = torch::cat({line_x_coord, line_z_coord},1);
        auto plane_xy = torch::cat({line_x_coord, line_y_coord},1);
        int num_samples = local_samples.sizes()[0];
        if(!params->train_params.plane_vec_only)
        {
            torch::Tensor vec_node_id = torch::zeros(num_samples, torch::TensorOptions(torch::kLong).device(local_samples.device()));
            auto global_coordinate_tmp = global_coordinate_inroi.reshape({-1,3});
            line_x_coord = 2. * (global_coordinate_tmp.slice(1,0,1)-params->octree_params.tree_optimizer_params.tree_roi_min(0))/(params->octree_params.tree_optimizer_params.tree_roi_max(0) - params->octree_params.tree_optimizer_params.tree_roi_min(0)) -1;
                // line_y_coord = global_coordinate_tmp.slice(1,1,2);
            // line_z_coord = 2. * (global_coordinate_tmp.slice(1,2,3)-params->octree_params.tree_optimizer_params.tree_roi_min(2))/(params->octree_params.tree_optimizer_params.tree_roi_max(2) - params->octree_params.tree_optimizer_params.tree_roi_min(2)) -1;
            line_z_coord = global_coordinate_tmp.slice(1,2,3);
            // printf("test line coord\n");
            // PrintTensorInfo(line_x_coord);
            // PrintTensorInfo(line_z_coord);

            // if(params->train_params.tensor_op == "cat")
            {
                auto mat_feat = torch::empty({num_samples, 0},
                                                        torch::TensorOptions(torch::kFloat).device(local_samples.device()));
                mat_feat = torch::cat({mat_feat, IndirectGridSample2D(tensor_plane_vector->forward( 0), local_node_id, plane_yz)}, 1);
                mat_feat = torch::cat({mat_feat, IndirectGridSample2D(tensor_plane_vector->forward( 1), local_node_id, plane_xz)}, 1);
                mat_feat = torch::cat({mat_feat, IndirectGridSample2D(tensor_plane_vector->forward( 2), local_node_id, plane_xy)}, 1);
                // auto vec_feat = torch::empty({num_samples, 0},
                //                                         torch::TensorOptions(torch::kFloat).device(local_samples.device()));
                // vec_feat = torch::cat({vec_feat, IndirectGridSample1D(tensor_line_vector->forward(0), vec_node_id, line_x_coord)}, 1);
                // vec_feat = torch::cat({vec_feat, IndirectGridSample1D(tensor_line_vector->forward(1), vec_node_id, line_y_coord)}, 1);
                // vec_feat = torch::cat({vec_feat, IndirectGridSample1D(tensor_line_vector->forward(2), vec_node_id, line_z_coord)}, 1);
                neural_features_inroi = mat_feat ;
            }
            // else if(params->train_params.tensor_op == "add")
            // {
            //     auto mat_feat = IndirectGridSample2D(tensor_plane_vector->forward( 0), local_node_id, plane_yz);
            //     auto vec_feat = IndirectGridSample1D(tensor_line_vector->forward(0), vec_node_id, line_x_coord);
            //     neural_features_inroi = mat_feat * vec_feat;
            //     mat_feat = IndirectGridSample2D(tensor_plane_vector->forward( 1), local_node_id, plane_xz);
            //     vec_feat = IndirectGridSample1D(tensor_line_vector->forward(1), vec_node_id, line_y_coord);
            //     neural_features_inroi += mat_feat * vec_feat;
            //     mat_feat = IndirectGridSample2D(tensor_plane_vector->forward( 2), local_node_id, plane_xy);
            //     vec_feat = IndirectGridSample1D(tensor_line_vector->forward(2), vec_node_id, line_z_coord);
            //     neural_features_inroi += mat_feat * vec_feat;
            // }
            // else
            // {
            //     throw std::runtime_error{"current tenosr support operation cat add"};
            // }

        }
        // std::cout << "in roi node " <<std::get<0>(at::_unique(node_id_inroi)) << std::endl;
        // PrintTensorInfo(local_samples);
        // PrintTensorInfo(local_node_id);
    }
    // PrintTensorInfo(neural_features_inroi);
    torch::Tensor neural_features_outroi;
    {
        if(!params->train_params.use_ground_truth_volume)
        {
            auto local_samples = tree->ComputeLocalSamples(global_coordinate_outroi, node_id_outroi);
            auto local_node_id = torch::index_select(node_active_prefix_sum_outroi, 0, node_id_outroi.reshape({-1}).contiguous());
            // PrintTensorInfo(local_samples);
            // PrintTensorInfo(local_node_id);
            neural_features_outroi = IndirectGridSample3D(explicit_grid_generator->grid_data, local_node_id, local_samples);
            // auto neural_featurestest = grid_sampler->forward(explicit_grid_generator->forward(local_node_id), local_samples);

            // PrintTensorInfo(neural_features_outroi);
            // PrintTensorInfo(neural_features_outroi- neural_featurestest);
            // char c = getchar();
        }

    }
    return {neural_features_inroi, neural_features_outroi};
}
torch::Tensor GeometryTensorQO::VolumeRegularizer()
{


    torch::Tensor tv_loss, edge_loss, zero_loss, nlcf_loss, l2_loss;
    torch::Tensor grid_plane_yz, grid_plane_xz, grid_plane_xy;
    torch::Tensor grid_line_x, grid_line_y, grid_line_z;

    {
        {
            auto local_node_id = torch::index_select(node_active_prefix_sum_inroi, 0, in_roi_node_index);
            // printf("test tv\n");
            // PrintTensorInfo(local_node_id);
            torch::Tensor vec_node_id = torch::zeros(1, torch::TensorOptions(torch::kLong).device(device));
            grid_plane_yz = torch::index_select(tensor_plane_vector->forward(0), 0, local_node_id);
            grid_plane_xz = torch::index_select(tensor_plane_vector->forward(1), 0, local_node_id);
            grid_plane_xy = torch::index_select(tensor_plane_vector->forward(2), 0, local_node_id);
            grid_line_x = torch::index_select(tensor_line_vector->forward(0).squeeze(-1), 0, vec_node_id);
            grid_line_y = torch::index_select(tensor_line_vector->forward(1).squeeze(-1), 0, vec_node_id);
            grid_line_z = torch::index_select(tensor_line_vector->forward(2).squeeze(-1), 0, vec_node_id);
        }
        torch::Tensor factor;
        TVLoss tv;
        if(params->train_params.use_loss_tv_3d)
        {
            torch::Tensor tv3d_node_id;
            if(tv_coord_global.defined())
            {
                tv3d_node_id = in_roi_node_index.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand({-1, tv_coord_global.size(1), tv_coord_global.size(2), tv_coord_global.size(3)});
                int tv3d_node_id_n = tv_coord_global.size(0);
                int tv3d_node_id_x = tv_coord_global.size(1);
                int tv3d_node_id_y = tv_coord_global.size(2);
                int tv3d_node_id_z = tv_coord_global.size(3);
                tv3d_node_id = tv3d_node_id.reshape({-1});
                // tv_coord_global = tv_coord_global.reshape({-1,3});

                auto neural_features = SampleVolumeIndirect_feature(tv_coord_global.reshape({-1,3}), tv3d_node_id);
                neural_features = neural_features.reshape({tv3d_node_id_n,tv3d_node_id_x,tv3d_node_id_y,tv3d_node_id_z,-1}).contiguous();
                neural_features = neural_features.permute({0, 4, 1, 2, 3}).contiguous();
                tv_loss = tv.forward(neural_features, factor) * params->train_params.loss_tv;
            }
            else if(tv_line_x_coord_global.defined())
            {
                tv3d_node_id = in_roi_node_index.expand({-1, tv_line_x_coord_global.size(1), tv_line_x_coord_global.size(2), tv_line_x_coord_global.size(3)});
            }
            else
            {
                throw std::runtime_error{"both tv3d coord not defined\n"};
            }

        }
        else
        {
            tv_loss = tv.forward(grid_plane_xz, factor) * params->train_params.loss_tv;
            tv_loss += tv.forward(grid_plane_yz, factor) * params->train_params.loss_tv;
            tv_loss += tv.forward(grid_plane_xy, factor) * params->train_params.loss_tv;
            if(!params->train_params.plane_vec_only)
            {
                // if(params->train_params.test_line_x > 0 && params->train_params.line_scale > 0 && params->train_params.test_linexz_scale >0 )
                // {
                //     tv_loss += tv.forward(tv_grid_line_x, factor)  * params->train_params.loss_tv * params->train_params.line_scale * params->train_params.test_linexz_scale;
                // }
                // if(params->train_params.test_line_y > 0 && params->train_params.line_scale > 0)
                // {
                //     tv_loss += tv.forward(tv_grid_line_y, factor)  * params->train_params.loss_tv * params->train_params.line_scale;
                // }
                // if(params->train_params.test_line_z > 0 && params->train_params.line_scale > 0 && params->train_params.test_linexz_scale >0 )
                // {
                //     tv_loss += tv.forward(tv_grid_line_z, factor)  * params->train_params.loss_tv * params->train_params.line_scale* params->train_params.test_linexz_scale;
                // }
                tv_loss += tv.forward(grid_line_x, factor)  * params->train_params.loss_tv * params->train_params.test_line_x;
                tv_loss += tv.forward(grid_line_y, factor)  * params->train_params.loss_tv * params->train_params.test_line_y;
                tv_loss += tv.forward(grid_line_z, factor)  * params->train_params.loss_tv * params->train_params.test_line_z;
            }

            if(params->train_params.test_para > 0)
            {
                auto local_node_id = torch::index_select(node_active_prefix_sum_outroi, 0, out_roi_node_index.reshape({-1}).contiguous());
                auto grid = explicit_grid_generator(local_node_id);
                tv_loss += tv.forward(grid, factor) * params->train_params.loss_tv;
            }
        }
    }

    if(params->train_params.loss_l2 > 0)
    {
        l2_loss = grid_plane_xz.square().mean() * params->train_params.loss_l2;
        l2_loss += grid_plane_yz.square().mean() * params->train_params.loss_l2;
        l2_loss += grid_plane_xy.square().mean() * params->train_params.loss_l2;
        if(!params->train_params.plane_vec_only)
        {
            l2_loss += grid_line_x.square().mean() * params->train_params.loss_l2;
            l2_loss += grid_line_y.square().mean() * params->train_params.loss_l2;
            l2_loss += grid_line_z.square().mean() * params->train_params.loss_l2;
        }
        // if(params->train_params.test_para > 0)
        {
            auto local_node_id = torch::index_select(node_active_prefix_sum_outroi, 0, out_roi_node_index.reshape({-1}).contiguous());
            auto grid = explicit_grid_generator(local_node_id);
            l2_loss += grid.square().mean();
        }

    }

    if(params->train_params.loss_edge > 0)
    {
        torch::Tensor neural_features;
        torch::Tensor neural_features_inroi, neural_features_outroi;
        if((!(neighbor_samples_yz.size() > 0)) || (!(neighbor_samples_xz.size()>0)) || (!(neighbor_samples_xy.size()>0)))
        {
            CHECK_GT(neighbor_samples.size(),0);
            #if 1
                std::tie (neural_features_inroi, neural_features_outroi) = SampleVolumeIndirect_edge(neighbor_samples.global_coordinate, neighbor_samples.node_id);
                auto t1 = neural_features_inroi.slice(0,0, neural_features_inroi.size(0),2);
                auto t2 = neural_features_inroi.slice(0,1, neural_features_inroi.size(0),2);
                // L2 Loss OK
                auto edge_error = ((t1-t2)).pow(2).mean(1).sqrt(); // set edge loss to 0.0001 is OK
                // auto edge_error = ((t1 -t2).abs().mean(1));
                // auto edge_error = ((t1-t2)*(torch::sqrt(t1*t1 + t2 * t2))).pow(2).mean(1);
                edge_loss = edge_error.mean() * params->train_params.loss_edge * neural_features_inroi.size(1);

                if(!params->train_params.use_ground_truth_volume)
                {
                    t1 = neural_features_outroi.slice(0,0,neural_features_outroi.size(0),2);
                    t2 = neural_features_outroi.slice(0,1,neural_features_outroi.size(0),2);
                    edge_error = ((t1-t2)).pow(2).mean(1);
                    edge_loss += edge_error.sqrt().mean() * params->train_params.loss_edge * neural_features_outroi.size(1);
                }


            #else
                neural_features = SampleVolumeIndirect(neighbor_samples.global_coordinate, neighbor_samples.node_id);
                auto t1 = neural_features.slice(0,0, neural_features.size(0),2);
                auto t2 = neural_features.slice(0,1, neural_features.size(0),2);
                // L2 Loss OK
                auto edge_error = ((t1-t2)).pow(2).mean(1); // set edge loss to 0.0001 is OK
                // auto edge_error = ((t1-t2)*(torch::sqrt(t1*t1 + t2 * t2))).pow(2).mean(1);
                edge_loss = edge_error.sqrt().mean() * params->train_params.loss_edge * neural_features.size(1);
            #endif


            // auto edge_error = (t1 - t2).abs().mean(1);
            // edge_loss = edge_error.mean() * params->train_params.loss_edge * neural_features.size(1);

            // edge_loss = (t1-t2).pow(2).mean() * params->train_params.loss_edge * neural_features.size(1);
        }
        // test edge
        else
        {
            CHECK_GT(neighbor_samples_yz.size(),0);
            CHECK_GT(neighbor_samples_xz.size(),0);
            CHECK_GT(neighbor_samples_xy.size(),0);

            torch::Tensor per_ray_sum;

            {
                int i = 0;
                int num_rays = neighbor_samples_yz.size();
                auto local_samples =  tree->ComputeLocalSamples(neighbor_samples_yz.global_coordinate, neighbor_samples_yz.node_id);
                auto local_node_id = tree->GlobalNodeIdToLocalActiveId(neighbor_samples_yz.node_id);
                auto torch_plane_xz_index = torch_plane_index.slice(0,i,i+1).squeeze(0);
                auto grid_plane_xz = tensor_plane_vector->forward(i);
                auto local_samples_xz = torch::index_select(local_samples, 1, torch_plane_xz_index);
                per_ray_sum = IndirectGridSample2D(grid_plane_xz, local_node_id, local_samples_xz);
                CHECK_EQ(num_rays, per_ray_sum.sizes()[0]);
                auto t1 = per_ray_sum.slice(0,0, per_ray_sum.size(0),2);
                auto t2 = per_ray_sum.slice(0,1, per_ray_sum.size(0),2);
                auto edge_error = ((t1-t2)).pow(2).mean(1).sqrt();
                if(edge_loss.defined())
                {
                    edge_loss += edge_error.mean() * params->train_params.loss_edge * per_ray_sum.size(1);
                }
                else
                {
                    // PrintTensorInfo(edge_error);
                    edge_loss = edge_error.mean() * params->train_params.loss_edge * per_ray_sum.size(1);
                }
            }
            {
                int i = 1;
                int num_rays = neighbor_samples_xz.size();
                auto local_samples =  tree->ComputeLocalSamples(neighbor_samples_xz.global_coordinate, neighbor_samples_xz.node_id);
                auto local_node_id = tree->GlobalNodeIdToLocalActiveId(neighbor_samples_xz.node_id);
                auto torch_plane_xz_index = torch_plane_index.slice(0,i,i+1).squeeze(0);
                auto grid_plane_xz = tensor_plane_vector->forward(i);
                auto local_samples_xz = torch::index_select(local_samples, 1, torch_plane_xz_index);
                per_ray_sum = IndirectGridSample2D(grid_plane_xz, local_node_id, local_samples_xz);
                CHECK_EQ(num_rays, per_ray_sum.sizes()[0]);
                auto t1 = per_ray_sum.slice(0,0, per_ray_sum.size(0),2);
                auto t2 = per_ray_sum.slice(0,1, per_ray_sum.size(0),2);
                auto edge_error = ((t1-t2)).pow(2).mean(1).sqrt();
                if(edge_loss.defined())
                {
                    edge_loss += edge_error.mean() * params->train_params.loss_edge * per_ray_sum.size(1);
                }
                else
                {
                    // PrintTensorInfo(edge_error);
                    edge_loss = edge_error.mean() * params->train_params.loss_edge * per_ray_sum.size(1);
                }
            }
            {
                int i = 2;
                int num_rays = neighbor_samples_xy.size();
                auto local_samples =  tree->ComputeLocalSamples(neighbor_samples_xy.global_coordinate, neighbor_samples_xy.node_id);
                auto local_node_id = tree->GlobalNodeIdToLocalActiveId(neighbor_samples_xy.node_id);
                auto torch_plane_xz_index = torch_plane_index.slice(0,i,i+1).squeeze(0);
                auto grid_plane_xz = tensor_plane_vector->forward(i);
                auto local_samples_xz = torch::index_select(local_samples, 1, torch_plane_xz_index);
                per_ray_sum = IndirectGridSample2D(grid_plane_xz, local_node_id, local_samples_xz);
                CHECK_EQ(num_rays, per_ray_sum.sizes()[0]);
                auto t1 = per_ray_sum.slice(0,0, per_ray_sum.size(0),2);
                auto t2 = per_ray_sum.slice(0,1, per_ray_sum.size(0),2);
                auto edge_error = ((t1-t2)).pow(2).mean(1).sqrt();
                if(edge_loss.defined())
                {
                    edge_loss += edge_error.mean() * params->train_params.loss_edge * per_ray_sum.size(1);
                }
                else
                {
                    // PrintTensorInfo(edge_error);
                    edge_loss = edge_error.mean() * params->train_params.loss_edge * per_ray_sum.size(1);
                }
            }

        }

    }
    auto opt = torch::nn::functional::GridSampleFuncOptions();
    opt = opt.padding_mode(torch::kBorder).mode(torch::kBilinear).align_corners(true);

    torch::Tensor fourier_loss;
    float filter_rate = params->train_params.loss_fourier_rate_w;
    // float filter_rate2 = 0.2;
    float filter_rate2 = params->train_params.loss_fourier_rate;

  

    if(params->train_params.loss_fourier > 0)
    {
        torch::Tensor fft_value, fft_xdim, fft_ydim;
        {
            auto densities = SampleVolumeIndirect(fourier_grid, fourier_node_id).reshape({fourier_size_x, fourier_size_y, fourier_size_z});
            densities = densities.permute({1,0,2});
            std::initializer_list<int64_t> y1 = {-2,-1};
            c10::ArrayRef<int64_t> fft_dim1(y1);
            int cx = densities.size(1)/2;
            int cy = densities.size(2)/2;
            int rh = std::max((int)(filter_rate * cy),1);
            int rw = std::max((int)(filter_rate * cx), 1);
            int rh1 = (int) (filter_rate2 * cy);
            int rw1 = (int) (filter_rate2 * cx);
            auto fft_img = torch::fft::fftshift(torch::fft::fftn(densities, {}, fft_dim1, "forward"), fft_dim1).abs();
            fft_xdim = fft_img.slice(1,cy-rh+1,cy+rh).mean(1);
            fft_ydim = fft_img.slice(2,cx-rw+1,cx+rw).mean(2);
            fft_value = fft_img.detach().clone().unsqueeze(0);
        }

        // 1 30 300 300
        auto circle_value = torch::empty({1,fft_value.size(1)}, torch::TensorOptions().device(device));
        torch::Tensor factor;
        {
            int ox = fft_value.size(2)/2;
            int oy = fft_value.size(3)/2;
            int total = fft_value.size(2)/2 ;

            for(int i = 1 ; i < total; i++)
            {
                int interpolate_num = 2 * pi<double>() * i;

                float interpolate_step = 360./interpolate_num;
                // printf("test interp %d %d\n",i, interpolate_num, interpolate_step);

                auto th = torch::arange(0,360, interpolate_step, torch::TensorOptions().device(device));
                auto xr1 = i * torch::cos(th/180. * pi<double>()) + ox;
                auto yr1 = i * torch::sin(th/180. * pi<double>()) + oy;
                auto grid = torch::meshgrid({xr1,yr1});
                torch::Tensor grid_interp = torch::stack({grid[1], grid[0]},-1);
                grid_interp = grid_interp/(fft_value.size(2)-1) * 2 -1.;

                auto test1 = torch::nn::functional::grid_sample(fft_value, grid_interp.unsqueeze(0), opt);
                // PrintTensorInfo(grid_interp);
                // // PrintTensorInfo(test1);
                // PrintTensorInfo(test1.mean({2,3}).squeeze(0));
                circle_value = torch::cat({circle_value, test1.mean({2,3})},0);


            }
            // printf("last value\n");
            // PrintTensorInfo(circle_value);
            // PrintTensorInfo(fft_value);
            circle_value = torch::cat({circle_value.flipud(), torch::zeros({1,fft_value.size(1)}, torch::TensorOptions().device(device)),
                                        circle_value}, 0);
            {
                torch::Tensor x = torch::arange({total},torch::TensorOptions().device(device));
                auto y = 1 - 1/(torch::exp(x- filter_rate2 * total) +1);
                factor = torch::cat({y.flipud(), torch::zeros({1}, torch::TensorOptions().device(device)),
                                    y},0);
                // PrintTensorInfo(factor)
            }
            circle_value = circle_value.permute({1,0});
            factor = factor.unsqueeze(0).repeat({circle_value.size(0),1});
        }
        // printf("mean value");
        // PrintTensorInfo(fft_xdim);
        // PrintTensorInfo(fft_ydim);
        // // PrintTensorInfo(fft_img.slice(1,cy-rh+1,cy+rh).mean(1));
        // // PrintTensorInfo(fft_img.slice(2,cx-rw+1,cx+rw).mean(2));
        // PrintTensorInfo(circle_value);
        // PrintTensorInfo(factor);
        auto mask_x = fft_xdim > circle_value;
        auto mask_y = fft_ydim > circle_value;

        #if 1
        if(fourier_loss.defined())
        {
            fourier_loss += (((torch::masked_select(fft_xdim, mask_x) - torch::masked_select(circle_value, mask_x)) * torch::masked_select(factor, mask_x)).mean()
                        + ((torch::masked_select(fft_ydim, mask_y) - torch::masked_select(circle_value, mask_y)) * torch::masked_select(factor, mask_y)).mean())
                        * params->train_params.loss_fourier;
        }
        else
        {
            fourier_loss = (((torch::masked_select(fft_xdim, mask_x) - torch::masked_select(circle_value, mask_x)) * torch::masked_select(factor, mask_x)).mean()
                        + ((torch::masked_select(fft_ydim, mask_y) - torch::masked_select(circle_value, mask_y)) * torch::masked_select(factor, mask_y)).mean())
                        * params->train_params.loss_fourier;

            // fourier_loss1 = (torch::masked_select((fft_xdim - circle_value) * factor, mask_x).mean()
            //                 + torch::masked_select((fft_ydim - circle_value) * factor, mask_y).mean()) * params->train_params.loss_fourier;
        }
        #else
        if(fourier_loss.defined())
        {
            fourier_loss += (((fft_xdim - circle_value) * factor).mean() + ((fft_ydim - circle_value) * factor).mean())
                        * params->train_params.loss_fourier;
        }
        else
        {
            fourier_loss = (((fft_xdim - circle_value) * factor).mean() + ((fft_ydim - circle_value) * factor).mean())
                        * params->train_params.loss_fourier;

            // fourier_loss1 = (torch::masked_select((fft_xdim - circle_value) * factor, mask_x).mean()
            //                 + torch::masked_select((fft_ydim - circle_value) * factor, mask_y).mean()) * params->train_params.loss_fourier;
        }
        #endif
        // PrintTensorInfo(circle_value);
        // PrintTensorInfo(circle_value.slice(0,0,1) - circle_value.slice(0,circle_value.size(0)-1, circle_value.size(0)));
        // PrintTensorInfo(factor.slice(0,0,1)- factor.slice(0, factor.size(0)-1, factor.size(0)));
    }


    torch::Tensor loss;
    if(edge_loss.defined())
    {
        // printf("perform edge loss\n");
        if(loss.defined())
            loss+= edge_loss;
        else
            loss =edge_loss;
    }
    if(tv_loss.defined())
    {
        // printf("perform tv loss\n");
        if(loss.defined())
            loss += tv_loss;
        else
            loss = tv_loss;
    }
    if(nlcf_loss.defined())
    {
        // printf("perform nlcf loss\n");
        if(loss.defined())
            loss += nlcf_loss;
        else
            loss = nlcf_loss;
    }
    if(l2_loss.defined())
    {
        if(loss.defined())
            loss += l2_loss;
        else
            loss = l2_loss;
    }
    if(fourier_loss.defined())
    {
        if(loss.defined())
        {
            loss += fourier_loss;
        }
        else
        {
            loss = fourier_loss;
        }
    }
    // return torch::Tensor();
    return loss;
}

void GeometryTensorQO::Compute_edge_nlm_samples(bool scale0_start)
{
    std::cout <<"==== Compute edge nlm samples =====" << std::endl;
    std::cout << "params->train_params.use_nlm_loss_density " << params->train_params.use_nlm_loss_density << std::endl;
    std::cout << "params->train_params.use_loss_tv_3d " << params->train_params.use_loss_tv_3d << std::endl;
    tree->to(device);
    int in_roi_node_size = 0;
    {
        {
            auto node_min = torch::index_select(tree->node_position_min, 0, tree->active_node_ids).to(torch::kCPU);
            auto node_max = torch::index_select(tree->node_position_max, 0, tree->active_node_ids).to(torch::kCPU);
            auto node_mid = node_min + node_max;

            auto active_node_id =  tree->active_node_ids.to(torch::kCPU);
            float* node_min_ptr = node_min.data_ptr<float>();
            float* node_max_ptr = node_max.data_ptr<float>();
            float* node_mid_ptr = node_mid.data_ptr<float>();

            std::vector<long> in_roi_index_vec, out_roi_index_vec;
            float epsilon1 = 0;
            if(params->octree_params.use_quad_tree_rep)
            epsilon1 = 1e-6;
            torch::Tensor in_roi_node;
            for(int i = 0; i < tree->NumActiveNodes(); ++i)
            {
                if(in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, params->octree_params.tree_optimizer_params.tree_roi_min,
                        params->octree_params.tree_optimizer_params.tree_roi_max, i, epsilon1))
                {
                    in_roi_index_vec.push_back(active_node_id.data_ptr<long>()[i]);
                    // in_roi_node_vec.push_back(i);
                    in_roi_node_size += 1;
                }
                else
                {
                    out_roi_index_vec.push_back(active_node_id.data_ptr<long>()[i]);
                }

                // in_roi_node = torch::from_blob(&in_roi_node_vec[0], {(long)in_roi_node_vec.size()}, torch::TensorOptions().dtype(torch::kLong))
                //                             .clone().to(device);
            }
            in_roi_node_index = torch::from_blob(&in_roi_index_vec[0], {(long)in_roi_index_vec.size()}, torch::TensorOptions().dtype(torch::kLong))
                                        .clone().to(device);
            out_roi_node_index = torch::from_blob(&out_roi_index_vec[0], {(long)out_roi_index_vec.size()}, torch::TensorOptions().dtype(torch::kLong))
                                        .clone().to(device);

            printf("total in roi node\n");
            PrintTensorInfo(in_roi_node_index);
            PrintTensorInfo(out_roi_node_index);
            // std::cout << in_roi_node_index << std::endl;
        }
        {
            auto node_active =  tree->node_active.to(torch::kCPU);
            int active_count_inroi = 0;
            int active_count_outroi = 0;
            std::vector<long> node_active_prefix_sum_vec_inroi, node_active_prefix_sum_vec_outroi;
            // std::vector<long> node_active_prefix_sum_vec;
            float epsilon1 = 0;
            if(params->octree_params.use_quad_tree_rep)
            epsilon1 = 1e-6;
            auto node_min = tree->node_position_min.to(torch::kCPU);
            auto node_max = tree->node_position_max.to(torch::kCPU);
            auto node_mid = node_min + node_max;

            float* node_min_ptr = node_min.data_ptr<float>();
            float* node_max_ptr = node_max.data_ptr<float>();
            float* node_mid_ptr = node_mid.data_ptr<float>();

            // int active_count = 0;
            for(int i = 0; i < tree->NumNodes(); ++i)
            {
                if(node_active.data_ptr<int>()[i] == 1)
                {
                    if(in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, params->octree_params.tree_optimizer_params.tree_roi_min,
                            params->octree_params.tree_optimizer_params.tree_roi_max, i, epsilon1))
                    {
                        node_active_prefix_sum_vec_inroi.push_back(active_count_inroi);
                        // in_roi_node_vec.push_back(active_count_inroi);
                        node_active_prefix_sum_vec_outroi.push_back(-1);
                        active_count_inroi++;
                    }
                    else
                    {
                        node_active_prefix_sum_vec_outroi.push_back(active_count_outroi);
                        node_active_prefix_sum_vec_inroi.push_back(-1);
                        active_count_outroi++;
                    }
                    // node_active_prefix_sum_vec.push_back(active_count);
                    // active_count++;
                }
                else
                {
                    node_active_prefix_sum_vec_inroi.push_back(-1);
                    node_active_prefix_sum_vec_outroi.push_back(-1);
                    // node_active_prefix_sum_vec.push_back(-1);
                }
            }
            node_active_prefix_sum_inroi = torch::from_blob(&node_active_prefix_sum_vec_inroi[0], {(long)node_active_prefix_sum_vec_inroi.size()}, torch::TensorOptions().dtype(torch::kLong))
                                    .clone().to(device);
            node_active_prefix_sum_outroi = torch::from_blob(&node_active_prefix_sum_vec_outroi[0], {(long)node_active_prefix_sum_vec_outroi.size()}, torch::TensorOptions().dtype(torch::kLong))
                                    .clone().to(device);
            // node_active_prefix_sum = torch::from_blob(&node_active_prefix_sum_vec[0], {(long)node_active_prefix_sum_vec.size()}, torch::TensorOptions().dtype(torch::kLong))
            //                         .clone().to(device);

            PrintTensorInfo(node_active_prefix_sum_inroi);
            PrintTensorInfo(node_active_prefix_sum_outroi);
            // PrintTensorInfo(node_active_prefix_sum);
        }

        // in_roi_node_index = torch::from_blob(&in_roi_index_vec[0], {(long)in_roi_index_vec.size()}, torch::TensorOptions().dtype(torch::kLong))
        //                                 .clone().to(device);
    }
    // if(scale0_start)
    // {
    //     printf("reset tensor parameters \n");
    //     // 1
    //     tensor_line_vector->reset();
    //     tensor_plane_vector->reset();
    //     // // 2
    //     explicit_grid_generator->reset();
    //     // // 3
    //     // // decoder->reset_module_parameters();
    //     // // 4
    //     ResetGeometryOptimizer();

    //     // PrintTensorInfo(tensor_plane_vector->plane_coef_yz);
    //     // PrintTensorInfo(tensor_plane_vector->plane_coef_xz);
    //     // PrintTensorInfo(tensor_plane_vector->plane_coef_xy);
    //     // PrintTensorInfo(tensor_line_vector->line_coef_x);
    //     // // PrintTensorInfo(tensor_line_vector->line_coef_x.slice(2,line_tv_index[0], line_tv_index[1]));
    //     // PrintTensorInfo(tensor_line_vector->line_coef_y);
    //     // PrintTensorInfo(tensor_line_vector->line_coef_z);
    //     // PrintTensorInfo(explicit_grid_generator->grid_data);
    //     // decoder->reset_module_parameters();
    // }

    if(params->train_params.use_loss_tv_3d && params->train_params.loss_tv > 0)
    {
        auto node_pos_min = torch::index_select(tree->node_position_min, 0, in_roi_node_index);
        auto node_pos_max = torch::index_select(tree->node_position_max, 0, in_roi_node_index);
        auto node_size = node_pos_max - node_pos_min;
        int x_size = vec_grid_size[0] * params->train_params.loss_tv_scale;
        int y_size = vec_grid_size[1] * params->train_params.loss_tv_scale;
        int z_size = vec_grid_size[2] * params->train_params.loss_tv_scale;
        auto x_coord = torch::linspace(-1,1, x_size, torch::TensorOptions().device(device));
        auto y_coord = torch::linspace(-1,1, y_size, torch::TensorOptions().device(device));
        auto z_coord = torch::linspace(-1,1, z_size, torch::TensorOptions().device(device));
        auto coord = torch::meshgrid({x_coord, y_coord, z_coord});
        auto coord_onecell = torch::cat({coord[0].unsqueeze(-1), coord[1].unsqueeze(-1), coord[2].unsqueeze(-1)},-1);
        // [20,10,20,3]
        auto coord_allcell = coord_onecell.unsqueeze(0).repeat({in_roi_node_size,1,1,1,1});

        // printf("test tv global\n");
        // PrintTensorInfo(node_size.unsqueeze(1).unsqueeze(1).unsqueeze(1));
        tv_coord_global = (coord_allcell + 1.f)/2.f * node_size.unsqueeze(1).unsqueeze(1).unsqueeze(1) + node_pos_min.unsqueeze(1).unsqueeze(1).unsqueeze(1);
        // PrintTensorInfo(tv_coord_global);
        #if 0
            tv_line_x_coord_global = tv_coord_global.slice(4,0,1);
            tv_line_z_coord_global = tv_coord_global.slice(4,2,3);
            tv_line_x_coord = coord_allcell.slice(4,0,1);
            tv_line_y_coord = coord_allcell.slice(4,1,2);
            tv_line_z_coord = coord_allcell.slice(4,2,3);

            // PrintTensorInfo(tree->ActiveNodeTensor());
            // PrintTensorInfo(in_roi_node_index);
            // PrintTensorInfo(tree->ActiveNodeTensor()- in_roi_node_index);
            // std::cout << tree->ActiveNodeTensor() << std::endl;
            // std::cout << in_roi_node_index << std::endl;
            // PrintTensorInfo(tv_line_x_coord_global);
            // PrintTensorInfo(tv_line_z_coord_global);
            // PrintTensorInfo(tv_line_x_coord);
            // PrintTensorInfo(tv_line_y_coord);
            // PrintTensorInfo(tv_line_z_coord);
            char c = getchar();
            (void) tv_coord_global;
        #endif

    }
    if(params->train_params.loss_fourier > 0)
    {
        int total_size = (params->net_params.grid_size-1) * std::pow(2, params->octree_params.tree_depth);
        auto roi_min = params->octree_params.tree_optimizer_params.tree_roi_min;
        auto roi_max = params->octree_params.tree_optimizer_params.tree_roi_max;
        int x_size = (int) std::round(total_size * (roi_max(0) - roi_min(0))/2.);
        int y_size = (int) std::round(total_size * (roi_max(1) - roi_min(1)-2 * params->train_params.loss_fourier_shift)/2.);
        int z_size = (int) std::round(total_size * (roi_max(2) - roi_min(2))/2.);

        x_size = (int) (params->train_params.loss_fourier_scale * x_size);
        y_size = (int) (params->train_params.loss_fourier_scale * params->train_params.loss_fourier_scale_y * y_size);
        z_size = (int) (params->train_params.loss_fourier_scale * z_size);

        if(x_size%2 == 0)
        {
            x_size += 1;
        }
        if(z_size%2 == 0)
        {
            z_size += 1;
        }
        auto x_coord = torch::linspace(roi_min(0),
                                        roi_max(0),
                                        x_size, torch::TensorOptions().device(device));
        auto y_coord = torch::linspace(roi_min(1)+ params->train_params.loss_fourier_shift,
                                        roi_max(1)-params->train_params.loss_fourier_shift,
                                        y_size, torch::TensorOptions().device(device));
        auto z_coord = torch::linspace(roi_min(2),
                                        roi_max(2),
                                        z_size, torch::TensorOptions().device(device));
        auto coord = torch::meshgrid({x_coord, y_coord, z_coord});

        fourier_grid = torch::cat({coord[0].unsqueeze(-1), coord[1].unsqueeze(-1), coord[2].unsqueeze(-1)},-1);

        printf("fourier grid is \n");
        PrintTensorInfo(fourier_grid);
        fourier_size_x = fourier_grid.size(0);
        fourier_size_y = fourier_grid.size(1);
        fourier_size_z = fourier_grid.size(2);
        fourier_grid = fourier_grid.reshape({-1,3});

        torch::Tensor weight;
        std::tie(fourier_node_id, weight) = tree->NodeIdForPositionGPU(fourier_grid);
        (void) weight;
        PrintTensorInfo(fourier_grid);

    }

    if(params->train_params.loss_edge > 0)
    {
        Eigen::Vector<int, -1> shape_v;
        shape_v.resize(D);
        for(int i = 0; i < D; ++i)
        {
            shape_v(i) = (int)(params->net_params.edge_grid_scale * vec_grid_size[i]);
        }
        std::cout << "edge shape " << shape_v(0) <<" " << shape_v(1) << " " << shape_v(2) << std::endl;
        #if 1
        neighbor_samples = tree->NodeNeighborSamplesPlane2D(shape_v, 0.001, 0, params->octree_params.tree_optimizer_params.tree_edge_roi_min, params->octree_params.tree_optimizer_params.tree_edge_roi_max,
                        params->octree_params.tree_optimizer_params.use_tree_roi, false);
        // neighbor_samples = tree->NodeNeighborSamplesPlane2D(shape_v, 0.001, 0, params->octree_params.tree_optimizer_params.tree_roi_min, params->octree_params.tree_optimizer_params.tree_roi_max,
        //                 params->octree_params.tree_optimizer_params.use_tree_roi, false);
        printf("use global neighbours");
        PrintTensorInfo(neighbor_samples.global_coordinate);
        #else
        neighbor_samples_yz = tree->NodeNeighborSamplesPlane2D(shape_v, 0.001, 0, params->octree_params.tree_optimizer_params.tree_edge_roi_min, params->octree_params.tree_optimizer_params.tree_edge_roi_max,
                        params->octree_params.tree_optimizer_params.use_tree_roi, params->octree_params.use_quad_tree_rep);
        neighbor_samples_xz = tree->NodeNeighborSamplesPlane2D(shape_v, 0.001, 1, params->octree_params.tree_optimizer_params.tree_edge_roi_min, params->octree_params.tree_optimizer_params.tree_edge_roi_max,
                        params->octree_params.tree_optimizer_params.use_tree_roi, params->octree_params.use_quad_tree_rep);
        neighbor_samples_xy = tree->NodeNeighborSamplesPlane2D(shape_v, 0.001, 2, params->octree_params.tree_optimizer_params.tree_edge_roi_min, params->octree_params.tree_optimizer_params.tree_edge_roi_max,
                        params->octree_params.tree_optimizer_params.use_tree_roi, params->octree_params.use_quad_tree_rep);
        printf("use tree neighbours ");
        PrintTensorInfo(neighbor_samples_yz.global_coordinate);
        PrintTensorInfo(neighbor_samples_xz.global_coordinate);
        PrintTensorInfo(neighbor_samples_xy.global_coordinate);

        #endif
    }

}

torch::Tensor GeometryTensorQO::Testcode(std::string ep_dir)
{
    // auto xyz_grid = nlm_xyz_grid.view({-1,3}).to(device);
    // auto node_id  = nlm_node_id.view({-1}).to(device);
    // auto density = SampleVolumeIndirect(xyz_grid, node_id);
    printf("tensor store info \n");
    PrintTensorInfo(tensor_plane_vector->plane_coef_yz);
    PrintTensorInfo(tensor_plane_vector->plane_coef_xz);
    PrintTensorInfo(tensor_plane_vector->plane_coef_xy);
    if(!params->train_params.plane_vec_only)
    {
        if(params->net_params.use_tree_line)
        {
            PrintTensorInfo(tensor_line_vector->line_coef_x);
            // PrintTensorInfo(tensor_line_vector->line_coef_x.slice(2,line_tv_index[0], line_tv_index[1]));
            PrintTensorInfo(tensor_line_vector->line_coef_y);
            PrintTensorInfo(tensor_line_vector->line_coef_z);
        }
        else
        {
            PrintTensorInfo(tensor_line_vector->line_coef_x);
            // PrintTensorInfo(tensor_line_vector->line_coef_x.slice(2,line_tv_index[0], line_tv_index[1]));
            PrintTensorInfo(tensor_line_vector->line_coef_y);
            PrintTensorInfo(tensor_line_vector->line_coef_z);

        }

    }

    // return density.to(torch::kCPU).view({nlm_node_id.size(0), nlm_node_id.size(1), nlm_node_id.size(2)});
    return torch::Tensor();
}

void GeometryTensorQO::SaveTensor(std::string ep_dir)
{
    auto matrix = torch::jit::pickle_save(tensor_plane_vector->plane_coef_yz);
    std::ofstream fout(ep_dir+"/tensor_matrix_yz.pt", std::ios::out | std::ios::binary);
    fout.write(matrix.data(), matrix.size());
    fout.close();
    fout.clear();

    matrix = torch::jit::pickle_save(tensor_plane_vector->plane_coef_xz);
    fout.open(ep_dir+"/tensor_matrix_xz.pt", std::ios::out | std::ios::binary);
    fout.write(matrix.data(), matrix.size());
    fout.close();
    fout.clear();

    matrix = torch::jit::pickle_save(tensor_plane_vector->plane_coef_xy);
    fout.open(ep_dir+"/tensor_matrix_xy.pt", std::ios::out | std::ios::binary);
    fout.write(matrix.data(), matrix.size());
    fout.close();
    fout.clear();

    auto line = torch::jit::pickle_save(tensor_line_vector->line_coef_x);
    fout.open(ep_dir+"/tensor_line_x.pt", std::ios::out | std::ios::binary);
    fout.write(line.data(), line.size());
    fout.close();
    fout.clear();

    line = torch::jit::pickle_save(tensor_line_vector->line_coef_y);
    fout.open(ep_dir+"/tensor_line_y.pt", std::ios::out | std::ios::binary);
    fout.write(line.data(), line.size());
    fout.close();
    fout.clear();

    line = torch::jit::pickle_save(tensor_line_vector->line_coef_z);
    fout.open(ep_dir+"/tensor_line_z.pt", std::ios::out | std::ios::binary);
    fout.write(line.data(), line.size());
    fout.close();
    fout.clear();
    if(params->train_params.loss_fourier > 0)
    {
        auto densities = SampleVolumeIndirect(fourier_grid, fourier_node_id).reshape({fourier_size_x, fourier_size_y, fourier_size_z});
        line = torch::jit::pickle_save(densities);
        fout.open(ep_dir+"/currentdensities.pt", std::ios::out | std::ios::binary);
        fout.write(line.data(), line.size());
        fout.close();
        fout.clear();
    }
}
