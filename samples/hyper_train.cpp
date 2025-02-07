﻿/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ColorizeTensor.h"
#include "saiga/vision/torch/ImageSimilarity.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/TrainParameters.h"

#include "Settings.h"
#include "data/Dataloader.h"
#include "geometry/all.h"
#include "modules/GradientCorrection.h"
#include "utils/utils.h"

#include "build_config.h"
#include "tensorboard_logger.h"
#include "utils/cimg_wrapper.h"
using namespace Saiga;

#ifdef UNUSED
#elif defined(__GNUC__)
# define UNUSED(x) UNUSED_ ## x __attribute__((unused))
#elif defined(__LCLINT__)
# define UNUSED(x) /*@unused@*/ x
#else
# define UNUSED(x) x
#endif

struct TrainScene
{
    std::shared_ptr<SceneBase> scene;
    HyperTreeBase tree = nullptr;
    std::shared_ptr<HierarchicalNeuralGeometry> neural_geometry;

    double last_eval_loss           = 9237643867809436;
    double new_eval_loss            = 9237643867809436;
    int last_structure_change_epoch = 0;

    void SaveCheckpoint(const std::string& dir)
    {
        auto prefix = dir + "/" + scene->scene_name + "_";
        scene->SaveCheckpoint(dir);

        torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
        torch::save(holder, prefix + "geometry.pth");

        torch::save(tree, prefix + "tree.pth");
    }

    void LoadCheckpoint(const std::string& dir)
    {
        auto prefix = dir + "/" + scene->scene_name + "_";
        scene->LoadCheckpoint(dir);

        std::cout << "Load checkpoint " << dir << std::endl;
        if (std::filesystem::exists(prefix + "geometry.pth"))
        {
            std::cout << "Load checkpoint geometry " << std::endl;
            torch::nn::ModuleHolder<HierarchicalNeuralGeometry> holder(neural_geometry);
            torch::load(holder, prefix + "geometry.pth");
        }
        std::cout << "file should be " << (prefix + "tree.pth") << std::endl;
        if (std::filesystem::exists(prefix + "tree.pth"))
        {
            std::cout << "load tree \n" << std::endl;
            torch::load(tree, prefix + "tree.pth");
            std::cout << "prefix is " << prefix << std::endl;
            std::cout << "now activte tree num is " << tree->NumActiveNodes() << std::endl;
        }
    }
};

class Trainer
{
   public:
    Trainer(std::shared_ptr<CombinedParams> params, std::string experiment_dir)
        : params(params), experiment_dir(experiment_dir)
    {
        torch::set_num_threads(4);
        torch::manual_seed(params->train_params.random_seed);

        tblogger = std::make_shared<TensorBoardLogger>((experiment_dir + "/tfevents.pb").c_str());

        for (auto scene_name : params->train_params.scene_name)
        {
            auto scene = std::make_shared<SceneBase>(params->train_params.scene_dir + "/" + scene_name);
            scene->train_indices =
                TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/train.txt");
            scene->test_indices =
                TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/eval.txt");

            // scene->input_angles =
            //     TrainParams::ReadIndexFile(scene->scene_path + "/" + params->train_params.split_name + "/eval.txt");

            if (params->train_params.train_on_eval)
            {
                std::cout << "Train on eval!" << std::endl;
                scene->test_indices = scene->train_indices;
            }

            std::cout << "Train(" << scene->train_indices.size() << "): " << array_to_string(scene->train_indices, ' ')
                      << std::endl;
            std::cout << "Test(" << scene->test_indices.size() << "): " << array_to_string(scene->test_indices, ' ')
                      << std::endl;

            scene->params = params;
            scene->LoadImagesCT(scene->train_indices);
            scene->LoadImagesCT(scene->test_indices);
            scene->Finalize();

            if (params->train_params.init_bias_with_bg)
            {
                scene->InitializeBiasWithBackground(tblogger.get());
            }
            scene->Draw(tblogger.get());


            scene->pose->AddNoise(params->train_params.noise_translation, params->train_params.noise_rotation);
            scene->camera_model->AddNoise(params->train_params.noise_intrinsics);

            auto prefix = params->train_params.checkpoint_directory + "/" + scene->scene_name + "_";
            auto tree   = HyperTreeBase(3, params->octree_params.tree_depth);
            std::cout <<"min max " << scene->dataset_params.roi_min(0) <<" "<< scene->dataset_params.roi_min(1) << " "<< scene->dataset_params.roi_min(2) <<std::endl;
            tree->settree(params->octree_params.use_quad_tree_rep, scene->dataset_params.roi_min(1), scene->dataset_params.roi_max(1));

            // // auto tree = HyperTreeBase(3, params->octree_params.tree_depth, params->octree_params.use_quad_tree_rep, params->octree_params.tree_roi_min(1), params->octree_params.tree_roi_max(1));
            // // auto tree = HyperTreeBase(3, params->octree_params.tree_depth, params->octree_params.use_quad_tree_rep, -0.3, 0.3);

            // std::vector<int> depthin;
            // depthin.push_back(3);
            // depthin.push_back(params->octree_params.tree_depth);
            // depthin.push_back(params->octree_params.use_quad_tree_rep);
            // std::vector<float> min_max;
            // min_max.push_back(params->octree_params.tree_roi_min(1));
            // min_max.push_back(params->octree_params.tree_roi_max(1));
            // auto tree = HyperTreeBase(depthin, min_max);

            tree->SetActive(params->octree_params.start_layer);

            // if not using intermidiate set the grid feature to be the same
            if(! params->net_params.using_fourier)
            {
                params->net_params.grid_features = params->net_params.intermidiate_features;
            }

            if(scene->ground_truth_volume.defined())
            params->train_params.use_ground_truth_volume = true;
            printf("grid features init %d \n", params->net_params.grid_features);
            TrainScene ts;

            {
                ts.neural_geometry = std::make_shared<GeometryTensorQO>(scene->num_channels, scene->D, tree, params);
            }


            ts.scene = scene;
            ts.tree  = tree;

            if(params->octree_params.tree_optimizer_params.optimize_tree_roi_at_ini && tree->NumNodes() > 1)
            {
                auto node_min = tree->node_position_min;
                auto node_max = tree->node_position_max;
                auto node_mid = node_min + node_max;

                std::cout << "before active node num " << tree->NumActiveNodes() << "num nodes " << tree->NumNodes()<< std::endl;
                auto new_culled_nodes = torch::zeros_like(tree->node_max_density).to(torch::kInt32);
                int * new_culled_nodes_ptr = new_culled_nodes.data_ptr<int>();
                auto active_nodes_id = tree->active_node_ids;
                std::cout << " region is " << scene->dataset_params.roi_min << std::endl << scene->dataset_params.roi_max << std::endl;

                std::cout << "tree roi is " << params->octree_params.tree_optimizer_params.tree_ini_roi_min << " " << params->octree_params.tree_optimizer_params.tree_ini_roi_max << std::endl;
                float epsilon = 0;
                if(params->octree_params.use_quad_tree_rep)
                epsilon = 1e-6;

                for(int i = 0; i < tree->NumActiveNodes();++i)
                {
                    int node_id = active_nodes_id.data_ptr<long>()[i];
                    float* node_min_ptr = node_min.data_ptr<float>()+ node_id * node_min.stride(0);
                    float* node_max_ptr = node_max.data_ptr<float>()+ node_id * node_max.stride(0);
                    float* node_mid_ptr = node_mid.data_ptr<float>()+ node_id * node_mid.stride(0);
                    // std::cout << "node id " << node_id << " " << node_min_ptr[0] << " " << node_min_ptr[1]  << " " << node_min_ptr[2] <<
                    //             " " << node_max_ptr[ 3] <<" " << node_max_ptr[1] << " " << node_max_ptr[2] << std::endl;
                    // if(!in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, scene->dataset_params.roi_min, scene->dataset_params.roi_max, node_id))

                    if(!in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, params->octree_params.tree_optimizer_params.tree_ini_roi_min, params->octree_params.tree_optimizer_params.tree_ini_roi_max, epsilon))
                    {
                        new_culled_nodes_ptr[node_id] = 1;
                    }
                }
                // char c = getchar();

                std::cout << "culled number " << new_culled_nodes.sum() << std::endl;

                tree->node_culled.add_(new_culled_nodes).clamp_(0,1);
                tree->node_active.add_(new_culled_nodes, -1).clamp_(0,1);
                tree->UpdateActive();
                tree->UpdateCulling();
            }
            std::cout << "now activte tree num is " << tree->NumActiveNodes() << std::endl;

            // std::string tree_dir6 = "/home/wangy0k/Desktop/owntree/hyperacorncontinue/Experiments/treedir6";
            // torch::save(tree, tree_dir6 + "/tree.pth");
            // printf("test here\n");
            ts.LoadCheckpoint(params->train_params.checkpoint_directory);
            ts.tree->to(device);

            // printf("to device \n");
            // ts.neural_geometry->ResetGeometryOptimizer();
            ts.neural_geometry->to(device);

            ts.neural_geometry->ResetGeometryOptimizer();

            ts.neural_geometry->Compute_edge_nlm_samples(false);
            scenes.push_back(ts);
        }
    }


    void Train()
    {
        int current_scale = params->train_params.current_scale;
        float downscale = pow(2, current_scale);
        for (auto& ts : scenes)
        {
            ts.scene->setcurrentproj(current_scale,ts.scene->train_indices);
            ts.scene->setcurrentproj(current_scale,ts.scene->test_indices);

        }

        int ori_output_volume_size = params->train_params.output_volume_size;
        int ori_volume_slice_num   = params->train_params.volume_slice_num;
        int ori_per_node_batch_size = params->train_params.per_node_batch_size;
        (void) ori_per_node_batch_size;
        params->train_params.output_volume_size = ori_output_volume_size/downscale;
        params->train_params.volume_slice_num   = ori_volume_slice_num/downscale;
        // params->train_params.per_node_batch_size = ori_per_node_batch_size/downscale;
        bool use_loss_nlm_vgg = params->train_params.use_loss_nlm_vgg;
        int ori_max_samples_per_node = params->octree_params.max_samples_per_node;
        params->octree_params.max_samples_per_node = ori_max_samples_per_node/downscale;
        std::cout << "current output volume size is " << params->train_params.output_volume_size << std::endl;
        float loss_tv = params->train_params.loss_tv;
        for (int epoch_id = 0; epoch_id <= params->train_params.num_epochs; ++epoch_id)
        {
            // enable larger volume & disable octree optimization
            if(epoch_id >= params->train_params.scale_0_iter)
            {
                params->train_params.use_loss_nlm_vgg = use_loss_nlm_vgg;
                // params->train_params.nlm_vgg_scale = 1/pow(2,current_scale);
                params->train_params.loss_tv = loss_tv;
            }
            else
            {
                params->train_params.loss_tv = loss_tv;
                // params->train_params.use_loss_nlm_vgg = false;
            }
            if(epoch_id == params->train_params.scale_0_iter)
            {
                current_scale = 0;
                for(auto& ts : scenes)
                {
                    ts.scene->setcurrentproj(current_scale,ts.scene->train_indices);
                    ts.scene->setcurrentproj(current_scale,ts.scene->test_indices);
                    ts.neural_geometry->Compute_edge_nlm_samples(epoch_id == params->train_params.scale_0_iter);
                }
                std::cout <<"compute samples " << (epoch_id == params->train_params.scale_0_iter) << std::endl;
                downscale = 1;
                params->train_params.output_volume_size = ori_output_volume_size/downscale;
                params->train_params.volume_slice_num   = ori_volume_slice_num/downscale;
                params->octree_params.max_samples_per_node = ori_max_samples_per_node/downscale;
                // params->train_params.per_node_batch_size = ori_per_node_batch_size/downscale;
                std::cout << "current output volume size is " << params->train_params.output_volume_size << std::endl;
                params->octree_params.optimize_structure = params->octree_params.optimize_structure_last;


            }


            // printf("epoch %d perform tv loss %B perform edge loss %B perform nlm vgg loss %B \n",
            //         (params->train_params.loss_tv > 0),
            //         (params->train_params.loss_edge > 0),
            //         (params->train_params.use_loss_nlm_vgg && params->train_params.loss_nlm_vgg > 0));
            if(epoch_id > params->train_params.num_epochs -5)
            {
                params->octree_params.optimize_structure = false;
                // if(params->train_params.test_para > 0)
                // {
                //     params->train_params.use_loss_nlm_vgg = false;
                // }
            }
            if(epoch_id > (params->train_params.num_epochs - params->train_params.nlm_stop))
            {
                params->train_params.use_loss_nlm_vgg = false;
            }

            if (epoch_id > params->train_params.num_zero_epochs)
            {
                params->train_params.loss_zero = 0;
            }

            std::cout << "\n==== Epoch " << epoch_id << " ====" << std::boolalpha
                        << " perform tv " << (params->train_params.loss_tv )
                        << " edge " << (params->train_params.loss_edge )
                        <<" nlm vgg " << params->train_params.vgg_loss_type <<" " << (params->train_params.use_loss_nlm_vgg && params->train_params.loss_nlm_vgg > 0)
                        <<" fourier " << (params->train_params.loss_fourier )
                        <<" nlm density " << params->train_params.use_nlm_loss_density
                        <<" use plane " << (params->train_params.plane_vec_only)
                        << " plane op " << (params->train_params.plane_op)
                        << std::endl;


            if(epoch_id == 0)
            {
                for (auto& ts : scenes)
                {
                    if (ts.tree->NumNodes() > 1)
                    {
                        auto std_value = Eval_cell_density(ts, ts.scene->train_indices, "Eval tv/" + ts.scene->scene_name, epoch_id);

                        auto active_node_id = ts.tree->ActiveNodeTensor();
                        auto std_value_select = torch::index_select(std_value, 0, active_node_id);
                        // torch::Tensor tv_value = torch::zeros_like(std_value_select);
                        // torch::Tensor nlm_value= torch::zeros_like(std_value_select);
                        torch::Tensor tv_value;
                        torch::Tensor nlm_value;
                        float loss_nlm_max = params->train_params.loss_nlm;
                        float loss_nlm_min = loss_nlm_max/params->train_params.loss_nlm_scale;

                        if(params->net_params.using_reverse_tv)
                        {
                            auto factor_a = (loss_nlm_max - loss_nlm_min)/(std_value_select.max() - std_value_select.min());
                            auto factor_b = loss_nlm_max - factor_a * std_value_select.max();

                            tv_value = std_value_select/std_value_select.max() * params->train_params.loss_tv;
                            // nlm_value = std_value_select/std_value_select.max() * params->train_params.loss_nlm;
                            nlm_value = factor_a * std_value_select + factor_b;

                        }
                        else
                        {

                            auto factor_a = (loss_nlm_min - loss_nlm_max)/(std_value_select.max() - std_value_select.min());
                            auto factor_b = loss_nlm_min - factor_a * std_value_select.max();

                            tv_value = std_value_select.min()/std_value_select * params->train_params.loss_tv;
                            // nlm_value = std_value_select.min()/std_value_select * params->train_params.loss_nlm;
                            nlm_value = factor_a * std_value_select + factor_b;

                        }
                        // auto tv_value = std_value_select/std_value_select.max() * params->train_params.loss_tv;
                        printf("std value is ");
                        PrintTensorInfo(std_value_select);
                        PrintTensorInfo(tv_value);

                        PrintTensorInfo(nlm_value);
                        ts.neural_geometry->setup_tv(tv_value);
                        ts.neural_geometry->setup_nlm(nlm_value);
                    }
                    else
                    {
                        torch::Tensor tv_value = torch::tensor(1.0);
                        torch::Tensor nlm_value = torch::tensor(1.0);
                        ts.neural_geometry->setup_tv(tv_value);
                        ts.neural_geometry->setup_nlm(nlm_value);
                    }

                }
            }

            bool checkpoint_it = epoch_id % params->train_params.save_checkpoints_its == 0 ||
                                 epoch_id == params->train_params.num_epochs;
            std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);
            std::string cp_str = checkpoint_it ? "Checkpoint(" + ep_str + ")" : "";
            std::string mom_str = "Mompoint(" + ep_str + ")";

            // std::cout << "\n==== Epoch " << epoch_id << " ====" << std::endl;
            if (epoch_id > 0)
            {
                // for (auto& ts : scenes)
                {
                    if(epoch_id > params->train_params.moment_start_epochs && params->train_params.use_moment)
                    {
                        for (auto& ts : scenes)
                        {
                            auto scene           = ts.scene;
                            Moment_cal(ts, scene->train_indices, "Mom/" + scene->scene_name, epoch_id,mom_str );
                        }
                    }
                    // auto scene = ts.scene;
                    TrainStep(epoch_id, true, "Train", false);
                    if (!params->train_params.train_on_eval && params->train_params.optimize_test_image_params)
                    {
                        TrainStep(epoch_id, false, "TestRefine", true);
                    }


                }
            }

            for (auto& ts : scenes)
            {
                auto scene           = ts.scene;
                auto tree            = ts.tree;
                auto neural_geometry = ts.neural_geometry;

                ComputeMaxDensity(ts);
                ts.last_eval_loss = ts.new_eval_loss;

                bool calculate_std = params->octree_params.optimize_structure &&
                                    epoch_id > params->train_params.optimize_tree_structure_after_epochs &&
                                    epoch_id < params->train_params.num_epochs &&
                                    (! (epoch_id < ts.last_structure_change_epoch + params->train_params.optimize_structure_every_epochs));
                ts.new_eval_loss =
                    EvalStepProjection(ts, scene->train_indices, "Eval/" + scene->scene_name, epoch_id, cp_str, false, calculate_std);
                if (!params->train_params.train_on_eval)
                {
                    EvalStepProjection(ts, scene->test_indices, "Test/" + scene->scene_name, epoch_id, cp_str, true, calculate_std);
                }
                // neural_geometry->PrintInfo();
                // scene->PrintInfo(epoch_id, tblogger.get());
            }


            if (epoch_id == 0 && params->octree_params.cull_at_epoch_zero)
            {
                std::cout << "Culling invisible nodes at epoch 0..." << std::endl;
                for (auto& ts : scenes)
                {
                    // ts.neural_geometry->SaveVolume(tblogger.get(), "culling_before", "", ts.scene->num_channels, 1,
                    //                                128);
                    NodeCulling(ts);
                    // ts.neural_geometry->SaveVolume(tblogger.get(), "culling_after", "", ts.scene->num_channels, 1,
                    // 128);
                }
            }

            if (checkpoint_it)
            {
                // torch::NoGradGuard ngg;
                auto ep_dir = experiment_dir + "ep" + ep_str + "/";
                std::cout << "Saving Checkpoint to " << ep_dir << std::endl;
                std::filesystem::create_directory(ep_dir);

                for (auto& ts : scenes)
                {
                    auto scene           = ts.scene;
                    auto tree            = ts.tree;
                    auto neural_geometry = ts.neural_geometry;
                    if (params->train_params.output_volume_size > 0)
                    {
                        int out_size = params->train_params.output_volume_size;
                        // if (epoch_id == params->train_params.num_epochs)
                        // change at last
                        if (epoch_id >= params->train_params.scale_0_iter)
                        {
                            // double resolution in last epoch
                            out_size *= params->train_params.output_volume_scale;
                            int mid_size = out_size;
                            if(params->octree_params.use_quad_tree_rep)
                            {
                                mid_size = (int)out_size * (scene->dataset_params.roi_max(1)- scene->dataset_params.roi_min(1))/2;

                            }
                            // std::cout <<"out put size " << out_size << std::endl;
                            // save last epoch as hdr image as well
                            std::cout << "saving volume as .hdr..." << std::endl;
                            auto [volume_density_crop, volume_node_id, volume_valid] = neural_geometry->UniformSampledVolume(
                                {out_size, mid_size, out_size}, scene->num_channels,scene->dataset_params.roi_min, scene->dataset_params.roi_max, true );
                            (void) volume_node_id, (void)volume_valid;
                            // int x_slice = volume_density.size(1);
                            // int z_slice = volume_density.size(3);

                            // auto volume_density_crop = volume_density;
                            if(!scene->ground_truth_volume.defined())
                            {

                                // volume_density_crop = volume_density_crop.slice(1, x_slice/4, 3 *x_slice/4).slice(3,z_slice/4, 3 *z_slice/4 );
                                if (scene->dataset_params.log_space_input)
                                {
                                    volume_density_crop *= scene->dataset_params.xray_max;
                                }
                                else
                                {
                                    // if(scene->dataset_params.use_log10_conversion)
                                    // {
                                    //     float factor = std::log10(scene->dataset_params.xray_max/scene->dataset_params.xray_min);
                                    //     volume_density_crop = std::exp10(- factor *volume_density_crop);

                                    // }
                                    // else
                                    {
                                        printf("take normal log transform\n");
                                        float factor = std::log(scene->dataset_params.xray_max/scene->dataset_params.xray_min);
                                        volume_density_crop = torch::exp(-factor * volume_density_crop);
                                    }
                                }
                            }
                            else
                            {
                                auto target      = ts.scene->ground_truth_volume;
                                auto target_max = target.max();
                                // if(false)
                                // {
                                std::cout << "save hdr img tensor " << TensorInfo(volume_density_crop) << " " << TensorInfo(target) << std::endl;
                                target = target/target_max;
                                volume_density_crop = volume_density_crop/target_max;
                                SaveHDRImageTensor(target, ep_dir + "/" + scene->scene_name + "_target_volume.hdr");
                            }
                            printf("save data shape\n");
                            // PrintTensorInfo(volume_density_crop);
                            std::cout << "volume density crop " << TensorInfo(volume_density_crop) << std::endl;
                            SaveHDRImageTensor(volume_density_crop, ep_dir + "/" + scene->scene_name + "_volume.hdr");

                            // auto density = neural_geometry->Testcode();
                            // float factor = std::log(scene->dataset_params.xray_max/scene->dataset_params.xray_min);
                            // density = torch::exp(-factor * density);
                            // SaveHDRImageTensor(density.unsqueeze(0), ep_dir + "/" + scene->scene_name + "_volume_small.hdr");

                            // PrintTensorInfo(volume_density_crop);

                            for(int img_tosave = -1; img_tosave < 3; ++img_tosave)
                            {
                                int slice_index = params->train_params.volume_slice_view + (img_tosave) * 10;
                                auto volume_slice = volume_density_crop.slice(2, slice_index, slice_index+1);

                                PrintTensorInfo(volume_slice);
                            // PrintTensorInfo(volume_slice.squeeze(2).squeeze(0).unsqueeze(0).unsqueeze(0));

                                auto im1 = TensorToImage<float> (volume_slice.squeeze(2).squeeze(0).unsqueeze(0).unsqueeze(0));
                                auto colorized = ImageTransformation::ColorizeTurbo(im1);
                                TemplatedImage<unsigned short> im1_new(im1.dimensions());
                                for(int i : im1.rowRange())
                                {
                                    for(int j : im1.colRange())
                                    {
                                        im1_new(i,j) = im1(i,j) * std::numeric_limits<unsigned short>::max();
                                    }
                                }
                                im1_new.save(ep_dir + "/" + scene->scene_name +  "imageslice_" + std::to_string(slice_index) + ".png");

                            }


                            // std::string output_final_roi = ep_dir + "/" + scene->scene_name + "_volume_roi.hdr";
                            // neural_geometry->SampleVolumeTest(output_final_roi);
                        }
                        else
                        {
                            neural_geometry->SaveVolume(tblogger.get(), cp_str + "/volume" + "/" + scene->scene_name, "",
                                                    scene->num_channels,
                                                    scene->dataset_params.vis_volume_intensity_factor, out_size,
                                                    params->train_params.output_dim, scene->dataset_params.roi_min, scene->dataset_params.roi_max);
                        }
                    }

                    if(checkpoint_it)
                    {
                        std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);
                        auto ep_dir = experiment_dir + "ep" + ep_str + "/";
                        std::filesystem::create_directory(ep_dir);
                        neural_geometry->SaveTensor(ep_dir);

                    }

                    ts.SaveCheckpoint(ep_dir);

                    if (scene->ground_truth_volume.defined())
                    {
                        EvalVolume(ts, cp_str + "/volume_gt" + "/" + scene->scene_name, epoch_id,
                                   epoch_id == params->train_params.num_epochs, ep_dir);
                    }
                }
            }

            // don't split before first epoch and after last epoch
            if (params->octree_params.optimize_structure &&
                epoch_id > params->train_params.optimize_tree_structure_after_epochs &&
                epoch_id < params->train_params.num_epochs)
            {
                for (auto& ts : scenes)
                {
                    if (epoch_id <
                        ts.last_structure_change_epoch + params->train_params.optimize_structure_every_epochs)
                    {
                        // has recently updated the structure
                        continue;
                    }

                    float converged_threshold = params->train_params.optimize_structure_convergence;
                    if (ts.new_eval_loss < ts.last_eval_loss * converged_threshold)
                    {
                        // not converged enough yet -> don't do structure change
                        std::cout << "Skip Structure Opt. (not converged) " << (float)ts.last_eval_loss << " -> "
                                  << (float)ts.new_eval_loss << " | " << float(ts.new_eval_loss / ts.last_eval_loss)
                                  << "<" << converged_threshold << std::endl;
                        continue;
                    }

                    auto scene           = ts.scene;
                    auto neural_geometry = ts.neural_geometry;
                    auto tree            = ts.tree;

                    if(!std::filesystem::is_directory(experiment_dir + "tmp"))
                    {
                        std::filesystem::create_directory(experiment_dir +"tmp");
                    }
                    tree->to(torch::kCPU);
                    std::cout << " save tree to " << experiment_dir +"tmp/tree.pth" <<"..." << std::endl;
                    torch::save(tree, experiment_dir +"tmp/tree.pth");
                    HyperTreeBase old_tree = HyperTreeBase(3, params->octree_params.tree_depth);

                    std::cout <<"min max " << scene->dataset_params.roi_min(0) <<" "<< scene->dataset_params.roi_min(1) << " "<< scene->dataset_params.roi_min(2) <<std::endl;
                    old_tree->settree(params->octree_params.use_quad_tree_rep, scene->dataset_params.roi_min(1), scene->dataset_params.roi_max(1));

                    torch::load(old_tree,experiment_dir +"tmp/tree.pth");


                    neural_geometry->SaveVolume(
                        tblogger.get(),
                        "StrucOpt/" + leadingZeroString(epoch_id, 4) + "/volume_before" + "/" + scene->scene_name, "",
                        scene->num_channels, scene->dataset_params.vis_volume_intensity_factor,
                        params->train_params.struct_opt_volume_size, params->train_params.output_dim, scene->dataset_params.roi_min, scene->dataset_params.roi_max);


                    if (params->octree_params.node_culling)
                    {
                        NodeCulling(ts);
                    }

                    // clone old tree structure


                    old_tree->to(torch::kCUDA);
                    OptimizeTreeStructure(ts, epoch_id);
                    neural_geometry->InterpolateInactiveNodes(old_tree);

                    neural_geometry->Compute_edge_nlm_samples(epoch_id >= params->train_params.scale_0_iter);

                    neural_geometry->SaveVolume(
                        tblogger.get(),
                        "StrucOpt/" + leadingZeroString(epoch_id, 4) + "/volume_after" + "/" + scene->scene_name, "",
                        scene->num_channels, scene->dataset_params.vis_volume_intensity_factor,
                        params->train_params.struct_opt_volume_size, params->train_params.output_dim, scene->dataset_params.roi_min,
                        scene->dataset_params.roi_max);

                    ts.last_structure_change_epoch = epoch_id;
                    ts.new_eval_loss               = 93457345345;
                    ts.last_eval_loss              = 93457345345;
                }
            }
        }
    }

   private:
    void ComputeMaxDensity(TrainScene& ts)
    {
        auto scene    = ts.scene;
        auto tree     = ts.tree;
        auto geometry = ts.neural_geometry;

        torch::NoGradGuard ngg;

        auto active_node_id = tree->ActiveNodeTensor();

        // Create for each node a 16^3 cube of samples
        // [num_nodes, 16, 16, 16, 3]
        torch::Tensor node_grid_position;
        if (params->net_params.geometry_type == "tensorfvm" || params->net_params.geometry_type=="tensorhash"|| params->net_params.geometry_type=="hash" )
        {
           node_grid_position = tree->UniformGlobalSamples(active_node_id, 20);
        }
        else
        {
            node_grid_position = tree->UniformGlobalSamples(active_node_id, params->net_params.std_grid_size);
        }
        int num_nodes           = node_grid_position.size(0);
        // printf("num_nodes %d\n", num_nodes);
        // printf("node grid positon\n");
        // PrintTensorInfo(node_grid_position);
        // [num_nodes, group_size, 3]
        node_grid_position = node_grid_position.reshape({num_nodes, -1, 3});
        // printf("max density 0.1\n");
        //        auto node_grid_mask = torch::ones({num_nodes, group_size, 1}, node_grid_position.options());
        auto node_grid_mask = scene->PointInAnyImage(node_grid_position).unsqueeze(2);
        // [num_nodes, group_size, 1]


        auto  density = geometry->SampleVolumeBatched(node_grid_position, node_grid_mask, active_node_id, params->net_params.using_decoder);

        //     char c = getchar();
        // [num_nodes]
        auto [per_node_max_density, max_index] = density.reshape({density.size(0), -1}).max(1);
        (void) max_index;

        auto tree_max_density = torch::zeros_like(tree->node_max_density);
        // tree_max_density.scatter_add_(0, tree->active_node_ids, per_node_max_density);

        // std::cout << per_node_max_density << std::endl;
        tree->node_max_density.index_copy_(0, tree->active_node_ids, per_node_max_density);
    }

    torch::Tensor ComputeStd(TrainScene& ts)
    {
        // auto scene      = ts.scene;
        auto tree       = ts.tree;
        auto geometry   = ts.neural_geometry;

        torch::NoGradGuard ngg;

        auto active_node_id = tree->ActiveNodeTensor();

        torch::Tensor densities = torch::empty({active_node_id.size(0), 0 });

        for(int i = 0; i < params->octree_params.std_iteration; i++)
        {
            // Create for each active node for a cube of samples
            // [num_nodes, 16, 16 ,16, 3]
            auto node_grid_position = tree->CreateSamplesRandomly(active_node_id, params->net_params.rand_grid_size);
            int num_nodes           = node_grid_position.size(0);
            // [num_nodes, groupsize, 3]
            node_grid_position = node_grid_position.reshape({num_nodes,-1,3});

            auto node_grid_mask = torch::ones({num_nodes, node_grid_position.size(1),1}, node_grid_position.options());
            // [num_nodes, group_size,1]
            torch::Tensor density;
            if (params->net_params.geometry_type == "exim")
            {
                density = geometry->SampleVolumeBatched(node_grid_position, node_grid_mask, active_node_id, false);
            }
            else
            {
                density = geometry->SampleVolumeBatched(node_grid_position, node_grid_mask, active_node_id, true);
            }

            densities = torch::cat({densities, density.squeeze(-1).to(torch::kCPU)},1);
        }

        // auto standard_dev = densities.std()
        auto standard_dev = torch::std(densities, 1,"correction");

        // auto standard_dev2 = torch::std(densities, 1,"aten::std");


        torch::Tensor std_dev = torch::zeros({(long)tree->NumNodes()}, device);

        // std_dev.index_select(0, active_node_id) = standard_dev.to(device);
        std_dev.index_add_(0, active_node_id, standard_dev.to(device));



        return std_dev;
    }

    void NodeCulling(TrainScene& ts)
    {
        {
            auto scene    = ts.scene;
            auto tree     = ts.tree;
            auto geometry = ts.neural_geometry;

            torch::NoGradGuard ngg;



            auto new_culled_nodes =
                ((tree->node_max_density < params->octree_params.culling_threshold) * tree->node_active)
                    .to(torch::kInt32).to(torch::kCPU);

            auto node_min = tree->node_position_min.to(torch::kCPU);
            auto node_max = tree->node_position_max.to(torch::kCPU);
            auto node_mid = node_min + node_max;
            // float* node_min_ptr = node_min.data_ptr<float>();
            // float* node_max_ptr = node_max.data_ptr<float>();
            // float* node_mid_ptr = node_mid.data_ptr<float>();

            int * new_culled_nodes_ptr = new_culled_nodes.data_ptr<int>();
            auto active_nodes_id = tree->active_node_ids.to(torch::kCPU);

            // printf("node culling feature\n");

            // PrintTensorInfo(node_min);
            // PrintTensorInfo(node_max);
            // PrintTensorInfo(node_mid);
            // PrintTensorInfo(new_culled_nodes);
            vec3 tree_roi_min = params->octree_params.tree_optimizer_params.tree_roi_min;
            vec3 tree_roi_max = params->octree_params.tree_optimizer_params.tree_roi_max;

            float epsilon = 0;
            if(params->octree_params.use_quad_tree_rep)
            epsilon = 1e-6;
            for(int i = 0; i < tree->NumActiveNodes();++i)
            {
                int node_id = active_nodes_id.data_ptr<long>()[i];
                float* node_min_ptr = node_min.data_ptr<float>()+ node_id * node_min.stride(0);
                float* node_max_ptr = node_max.data_ptr<float>()+ node_id * node_min.stride(0);
                float* node_mid_ptr = node_mid.data_ptr<float>()+ node_id * node_min.stride(0);
                if(in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, tree_roi_min, tree_roi_max, epsilon))
                {
                    new_culled_nodes_ptr[node_id] = 0;
                }
            }
            new_culled_nodes = new_culled_nodes.to(device);
            int num_culled_nodes = new_culled_nodes.sum().item().toInt();
            std::cout << "Culling " << num_culled_nodes << " nodes" << std::endl;
            PrintTensorInfo(new_culled_nodes);
            if (num_culled_nodes > 0)
            {
                // Integrate new culling into the tree
                tree->node_culled = tree->node_culled.to(device);
                tree->node_culled.add_(new_culled_nodes).clamp_(0, 1);
                tree->node_active.add_(new_culled_nodes, -1).clamp_(0, 1);
                tree->UpdateActive();
                tree->UpdateCulling();

                if (params->train_params.reset_optimizer_after_structure_change)
                {
                    std::cout << "Resetting Optimizer..." << std::endl;
                    geometry->ResetGeometryOptimizer();
                    current_lr_factor = 1;
                }
            }
        }
    }



    void TrainStep(int epoch_id, bool train_indices, std::string name, bool only_image_params)
    {
        std::vector<std::vector<int>> indices_list;
        std::vector<std::shared_ptr<SceneBase>> scene_list;

        // torch::autograd::AnomalyMode::set_enabled(true);

        // for(auto &ts : scenes)
        // {
        //     auto neural_geometry = ts.neural_geometry;
        //     std::vector<double> learningrate = neural_geometry->getDecoderLearningRate();
        //     std::cout <<"current learning rate " << learningrate[0] << std::endl;
        // }



        for (auto& ts : scenes)
        {
            auto scene           = ts.scene;
            auto tree            = ts.tree;
            auto neural_geometry = ts.neural_geometry;
            auto indices         = train_indices ? scene->train_indices : scene->test_indices;

            indices_list.push_back(indices);
            scene_list.push_back(scene);


            neural_geometry->train(epoch_id, true);
            // if(params->train_params.use_loss_nlm_vgg && epoch_id == params->train_params.scale_0_iter)
            // {
            //     neural_geometry->reset_vgg_parameters();
            // }
            if(params->net_params.geometry_type == "tensorquad" || params->net_params.geometry_type == "tensorfvm")
            {
                bool checkpoint_it = epoch_id % params->train_params.save_checkpoints_its == 0 ||
                                 epoch_id == params->train_params.num_epochs;
                std::string ep_str = Saiga::leadingZeroString(epoch_id, 4);
                auto ep_dir = experiment_dir + "ep" + ep_str + "/";
                // std::filesystem::create_directory(ep_dir);

                auto test = neural_geometry->Testcode(ep_dir);

            }
            scene->train(true);
        }

        auto options = torch::data::DataLoaderOptions()
                           .batch_size(params->train_params.batch_size)
                           .drop_last(false)
                           .workers(params->train_params.num_workers_train);
        //        auto dataset = RandomRaybasedSampleDataset(
        //            indices, scene, params, params->train_params.rays_per_image * (only_image_params ? 0.25 : 1));
        auto dataset     = RandomMultiSceneDataset(indices_list, scene_list, params,
                                                   params->train_params.rays_per_image * (only_image_params ? 0.25 : 1));
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(dataset, options);



        // optimize network with fixed structure
        Saiga::ProgressBar bar(std::cout, name + " " + std::to_string(epoch_id) + " |", dataset.size().value(), 10,
                               false, 1000, "ray");


        float epoch_loss_train  = 0;
        int processed_ray_count = 0;
        std::vector<double> batch_loss;

        for (SampleData sample_data : (*data_loader))
        {
            auto& scene           = scenes[sample_data.scene_id].scene;
            auto& tree            = scenes[sample_data.scene_id].tree;
            auto& neural_geometry = scenes[sample_data.scene_id].neural_geometry;

            RayList rays;
            if(params->train_params.use_defocus)
            {
                rays =
                    scene->GetRays(sample_data.pixels.uv, sample_data.pixels.image_id, sample_data.pixels.camera_id, sample_data.pixels.defocus);
            }
            else
            {
                rays =
                    scene->GetRays(sample_data.pixels.uv, sample_data.pixels.image_id, sample_data.pixels.camera_id);
            }


            SampleList all_samples;
            if(params->octree_params.tree_depth == 0)
            {
                all_samples = tree->CreateSamplesForRays_area(rays, params->octree_params.max_samples_per_node,
                                                                    params->train_params.train_interval_jitter,
                                                                    scene->dataset_params.roi_min,
                                                                    scene->dataset_params.roi_max);
            }
            else
            {
                all_samples = tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node,
                                                                    params->train_params.train_interval_jitter);
            }


            if (params->train_params.gradient_correction)
            {
                all_samples.global_coordinate =
                    GradientCorrection(all_samples.global_coordinate, rays.direction, all_samples.ray_index);
            }



            auto  [predicted_image, ratio, ratio_inv] =
                neural_geometry->ComputeImage(all_samples, scene->num_channels, sample_data.NumPixels(), params->net_params.using_decoder);
            (void) ratio_inv;
            // printf("test predicted image\n");
            // PrintTensorInfo(predicted_image);

            // printf("test here 00\n");
            predicted_image =
                scene->tone_mapper->forward(predicted_image, sample_data.pixels.uv, sample_data.pixels.image_id);
            // printf("test here 01\n");

            if (sample_data.pixels.target_mask.defined())
            {
                // Multiply by mask so the loss of invalid pixels is 0
                // printf("target mask defined.\n");
                // PrintTensorInfo(sample_data.pixels.target_mask);
                predicted_image           = predicted_image * sample_data.pixels.target_mask;
                sample_data.pixels.target = sample_data.pixels.target * sample_data.pixels.target_mask;
            }

            // printf("test here 02\n");

            CHECK_EQ(predicted_image.sizes(), sample_data.pixels.target.sizes());
            torch::Tensor per_ray_loss_mse, per_ray_loss_l1;
            if(ratio.defined())
            {
                // if(params->train_params.use_density_ratio)
                {
                    if(params->train_params.loss_weighted < 0)
                    {
                        per_ray_loss_mse =
                            ((predicted_image - sample_data.pixels.target*ratio) ).square();

                    }
                    else if(params->train_params.loss_weighted >= 0 && params->train_params.loss_weighted <10 )
                    {
                        auto predicted_value = predicted_image.detach().clone();
                        auto weighted = 1/((predicted_value + 1e-3).square());
                        if (sample_data.pixels.target_mask.defined())
                        {
                            weighted = weighted * sample_data.pixels.target_mask;
                        }
                        per_ray_loss_mse = weighted * ((predicted_image - sample_data.pixels.target*ratio) ).square();

                    }
                    else if(params->train_params.loss_weighted >= 10)
                    {
                        auto predicted_value = predicted_image.detach().clone();
                        auto weighted = ((predicted_value + 1e-3).square());
                        if (sample_data.pixels.target_mask.defined())
                        {
                            weighted = weighted * sample_data.pixels.target_mask;
                        }
                        per_ray_loss_mse = weighted * ((predicted_image - sample_data.pixels.target*ratio) ).square();
                    }
                    else
                    {
                        throw std::runtime_error{"not supported loss"};

                    }

                }
                // else
                // {
                //     per_ray_loss_mse =
                //         ((predicted_image/ratio_inv - sample_data.pixels.target) ).square();
                // }

            }
            else
            {
                // per_ray_loss_mse =
                //     ((predicted_image - sample_data.pixels.target) ).square();

                if(params->train_params.loss_weighted < 0)
                {
                    per_ray_loss_mse =
                    ((predicted_image - sample_data.pixels.target) ).square();
                }
                else if(params->train_params.loss_weighted >= 0 && params->train_params.loss_weighted <10 )
                {
                    auto predicted_value = predicted_image.detach().clone();
                    auto weighted = 1/((predicted_value + 1e-3).square());
                    if (sample_data.pixels.target_mask.defined())
                    {
                        weighted = weighted * sample_data.pixels.target_mask;
                    }
                    per_ray_loss_mse = weighted * ((predicted_image - sample_data.pixels.target) ).square();
                }
                else if(params->train_params.loss_weighted >= 10)
                {
                    auto predicted_value = predicted_image.detach().clone();
                    auto weighted = ((predicted_value + 1e-3).square());
                    if (sample_data.pixels.target_mask.defined())
                    {
                        weighted = weighted * sample_data.pixels.target_mask;
                    }
                    per_ray_loss_mse = weighted * ((predicted_image - sample_data.pixels.target) ).square();
                }
                else
                {
                    throw std::runtime_error{"not supported loss"};
                }
            }
            if(ratio.defined())
            {
                // if(params->train_params.use_density_ratio)
                {
                    per_ray_loss_l1 =
                        ((predicted_image - sample_data.pixels.target * ratio) ).abs();
                }
                // else
                // {
                //     per_ray_loss_l1 =
                //         ((predicted_image/ratio_inv - sample_data.pixels.target * ratio) ).abs();
                // }

            }
            else
            {
                per_ray_loss_l1 =
                    ((predicted_image - sample_data.pixels.target) ).abs();
            }

            auto per_ray_loss = (per_ray_loss_mse * params->train_params.loss_mse_scale +
                                 per_ray_loss_l1 * params->train_params.loss_l1_scale) *
                                params->train_params.loss_scale;

            auto avg_per_image_loss = (per_ray_loss).mean();
            auto volume_loss = neural_geometry->VolumeRegularizer();

            auto param_loss  = scene->tone_mapper->ParameterLoss(scene->active_train_images);
            // printf("test here 05\n");
            // auto total_loss  = avg_per_image_loss + param_loss;
            auto total_loss = avg_per_image_loss + param_loss;
            if (volume_loss.defined())
            {
                total_loss += volume_loss;
            }
            // printf("test here 5\n");

            static int global_batch_id_a = 0;
            static int global_batch_id_b = 0;
            int& global_batch_id         = only_image_params ? global_batch_id_b : global_batch_id_a;

            // printf("test here 6\n");
            // PrintTensorInfo(total_loss);
            total_loss.backward();
            // printf("test here 7\n");
            if (!only_image_params)
            {
                neural_geometry->PrintGradInfo(global_batch_id, tblogger.get());
                neural_geometry->OptimizerStep(epoch_id);
            }
            scene->PrintGradInfo(global_batch_id, tblogger.get());
            scene->OptimizerStep(epoch_id, only_image_params);

            float avg_per_image_loss_float = avg_per_image_loss.item().toFloat();
            tblogger->add_scalar("Loss" + name + "/" + scene->scene_name + "/batch", global_batch_id,
                                 avg_per_image_loss_float);
            batch_loss.push_back(avg_per_image_loss_float);
            // tblogger->add_scalar("TrainLoss/param", global_batch_id, param_loss.item().toFloat());
            epoch_loss_train += avg_per_image_loss_float * sample_data.NumPixels();
            processed_ray_count += sample_data.NumPixels();

            // float param_loss_float = param_loss.item().toFloat();
            float regularizer_loss = 0;
            if (volume_loss.defined())
            {
                regularizer_loss = volume_loss.item().toFloat();
            }

            bar.SetPostfix(" Cur=" + std::to_string(epoch_loss_train / processed_ray_count) +
                           // " Param: " + std::to_string(param_loss_float) +
                           " Reg: " + std::to_string(regularizer_loss));
            bar.addProgress(sample_data.NumPixels());

            global_batch_id++;
        }

        if (!only_image_params)
        {
            std::ofstream strm(experiment_dir + "/batch_loss.txt", std::ios_base::app);
            for (auto d : batch_loss)
            {
                strm << d << "\n";
            }
        }

        tblogger->add_scalar("TrainLoss/lr_factor", epoch_id, current_lr_factor);
        current_lr_factor *= params->train_params.lr_decay_factor;

        if(epoch_id == params->train_params.scale_0_iter)
        {
            for (auto& ts : scenes)
            {
                auto scene           = ts.scene;
                auto tree            = ts.tree;
                auto neural_geometry = ts.neural_geometry;
                neural_geometry->UpdateDecoderLearningRate(params->net_params.decoder_lr_decay);
            }
        }
        for (auto& ts : scenes)
        {
            auto scene           = ts.scene;
            auto tree            = ts.tree;
            auto neural_geometry = ts.neural_geometry;
            neural_geometry->UpdateLearningRate(params->train_params.lr_decay_factor);
            scene->UpdateLearningRate(params->train_params.lr_decay_factor);
        }
        tblogger->add_scalar("Loss" + name + "/total", epoch_id, epoch_loss_train / processed_ray_count);
    }

    void EvalVolume(TrainScene& ts, std::string tbname, int epoch_id, bool write_gt, std::string epoch_dir)
    {
        std::string name = ts.scene->scene_name;
        auto target      = ts.scene->ground_truth_volume;
        if(target.size(2) == target.size(3))
        {
            target      = ts.scene->ground_truth_volume.slice(2,ts.scene->dataset_params.slice_min, ts.scene->dataset_params.slice_max);
        }
        // auto target      = ts.scene->ground_truth_volume.slice(2,ts.scene->dataset_params.slice_min, ts.scene->dataset_params.slice_max);

        auto [volume, volume_node_id, volume_valid] = ts.neural_geometry->UniformSampledVolume(
            {target.size(1), target.size(2), target.size(3)}, ts.scene->num_channels, ts.scene->dataset_params.roi_min,
            ts.scene->dataset_params.roi_max, true);
        (void) volume_node_id, (void) volume_valid;
        auto target_max = target.max();
        // if(false)
        // {
        volume = volume.slice(2,ts.scene->dataset_params.slice_min, ts.scene->dataset_params.slice_max);
        target = target/target_max;
        volume = volume/target_max;
        // }


        // target = target.to(torch::kCUDA);
        // volume = volume.to(torch::kCUDA);
        std::cout << "gt:  " << TensorInfo(target) << std::endl;
        std::cout << "rec: " << TensorInfo(volume) << std::endl;

        float epoch_loss_train_ssim = 0;
        float epoch_loss_train_psnr = 0;
        PSNR psnr(0, 5);
        auto volume_error_psnr = psnr->forward(volume, target);
        epoch_loss_train_psnr = volume_error_psnr.item().toFloat();

        if(epoch_id == params->train_params.num_epochs)
        // if(epoch_id > 0 && (epoch_id % params->train_params.save_checkpoints_its == 0 ||
        //                          epoch_id == params->train_params.num_epochs))
        {
            SSIM3D ssim(5, 1);
            auto volume_ssim = ssim->forward(volume.unsqueeze(0), target.unsqueeze(0));
            epoch_loss_train_ssim = volume_ssim.item().toFloat();
        }
        // SSIM3D ssim(5, 1);
        // auto volume_ssim = ssim->forward(volume.unsqueeze(0), target.unsqueeze(0));
        // float epoch_loss_train_ssim = volume_ssim.item().toFloat();



        {
            std::ofstream loss_file(experiment_dir + "/psnr_ssim.txt", std::ios_base::app);
            loss_file << epoch_loss_train_psnr << "," << epoch_loss_train_ssim << "\n";
            loss_file << epoch_loss_train_psnr << ""  << "\n";

        }

        tblogger->add_scalar("Loss" + name + "/psnr", epoch_id, epoch_loss_train_psnr);
        tblogger->add_scalar("Loss" + name + "/ssim", epoch_id, epoch_loss_train_ssim);



        std::cout << ConsoleColor::GREEN << "> Volume Loss " << name << " | SSIM " << epoch_loss_train_ssim << " PSNR "
                  << epoch_loss_train_psnr << ConsoleColor::RESET << std::endl;
        // std::cout << ConsoleColor::GREEN << "> Volume Loss " << name  << " PSNR "
        //           << epoch_loss_train_psnr << ConsoleColor::RESET << std::endl;
    }

    void Moment_cal(TrainScene& ts, std::vector<int> indices, std::string name, int epoch_id,
                              std::string checkpoint_name)
    {
        auto scene              = ts.scene;
        auto tree               = ts.tree;
        auto neural_geometry    = ts.neural_geometry;

        neural_geometry->train(epoch_id, false);
        scene->train(false);
        torch::NoGradGuard ngg;

        int out_w = scene->cameras.front().w;
        int out_h = scene->cameras.front().h;

        int rows_per_batch = std::max(params->train_params.batch_size / out_w, 1);

        auto options = torch::data::DataLoaderOptions().batch_size(1).drop_last(false).workers(
            params->train_params.num_workers_eval);
        auto dataset     = RowRaybasedSampleDataset(indices, scene, params, out_w, out_h, rows_per_batch);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset, options);

        std::vector<torch::Tensor> projection_images(scene->frames.size());
        // std::vector<torch::Tensor> target_images(scene->frames.size());

        for(int i : indices)
        {
            projection_images[i] = torch::zeros({1, out_h, out_w});
            // target_images[i]     = torch::zeros({1, out_h, out_w});
        }

        // optimze network with fixed structure
        {
        // optimize network with fixed structure
        Saiga::ProgressBar bar(std::cout,
                            name + " (" + std::to_string(out_w) + "x" + std::to_string(out_h) + ") " +
                                std::to_string(epoch_id) + " |",
                            out_w * out_h * indices.size(), 10, false, 1000, "ray");

            for (RowSampleData sample_data : (*data_loader))
            {
                RayList rays =
                    scene->GetRays(sample_data.pixels.uv, sample_data.pixels.image_id, sample_data.pixels.camera_id);

                SampleList all_samples;
                if(params->octree_params.tree_depth == 0)
                {
                all_samples = tree->CreateSamplesForRays_area(rays, params->octree_params.max_samples_per_node,
                                                                    false,
                                                                    scene->dataset_params.roi_min,
                                                                    scene->dataset_params.roi_max);
                }
                else
                {
                    all_samples =
                    tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, false);
                }

                // SampleList all_samples =
                //     tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, false);

                auto    [predicted_image, ratio, ratio_inv] =
                    neural_geometry->ComputeImage(all_samples, scene->num_channels, sample_data.NumPixels(), params->net_params.using_decoder);

                (void) ratio;
                predicted_image =
                    scene->tone_mapper->forward(predicted_image, sample_data.pixels.uv, sample_data.pixels.image_id);



                if (sample_data.pixels.target_mask.defined())
                {
                    // Multiply by mask so the loss of invalid pixels is 0
                    if(ratio_inv.defined())
                    {
                        predicted_image           = predicted_image * sample_data.pixels.target_mask * ratio_inv;
                    }
                    else
                    {
                        predicted_image           = predicted_image * sample_data.pixels.target_mask;
                    }
                    sample_data.pixels.target = sample_data.pixels.target * sample_data.pixels.target_mask;
                }

                CHECK_EQ(predicted_image.sizes(), sample_data.pixels.target.sizes());

                // CHECK_EQ(image_samples.image_index.size(0), 1);
                int image_id = sample_data.image_id;

                if (projection_images[image_id].is_cpu())
                {
                    for (auto& i : projection_images)
                    {
                        if (i.defined()) i = i.cpu();
                    }
                    projection_images[image_id] = projection_images[image_id];
                }
                auto prediction_rows = predicted_image.reshape({predicted_image.size(0), -1, out_w});

                projection_images[image_id].slice(1, sample_data.row_start, sample_data.row_end) = prediction_rows;
                // bar.SetPostfix(" MSE=" + std::to_string(epoch_loss_train_mse / image_count));
                bar.addProgress(sample_data.NumPixels());
            }

        }


        std::string save_file_name = experiment_dir + "/" + scene->scene_name + "_reprojection_" + to_string(epoch_id) ;
        float factor = 1/(epoch_id-params->train_params.moment_start_epochs +1);
        scene->setmoment(projection_images, indices, factor, save_file_name);

        std::cout << "update moment finished" << std::endl;
        // return projection_images;

    }

    double EvalStepProjection(TrainScene& ts, std::vector<int> indices, std::string name, int epoch_id,
                              std::string checkpoint_name, bool test, bool calculate_std)
    {
        auto scene           = ts.scene;
        auto tree            = ts.tree;
        auto neural_geometry = ts.neural_geometry;

        neural_geometry->train(epoch_id, false);
        scene->train(false);
        torch::NoGradGuard ngg;

        double epoch_loss_train_l1  = 0;
        double epoch_loss_train_mse = 0;
        int image_count             = 0;

        // bool mult_error = false;

        float eval_scale = params->train_params.eval_scale;
        if(epoch_id == params->train_params.num_epochs && params->train_params.save_temp)
        {
            eval_scale = 1.0f;
        }

        int out_w = scene->cameras.front().w * eval_scale;
        int out_h = scene->cameras.front().h * eval_scale;

        int rows_per_batch = std::max(params->train_params.batch_size / out_w/4, 1);


        auto options = torch::data::DataLoaderOptions().batch_size(1).drop_last(false).workers(
            params->train_params.num_workers_eval);
        auto dataset     = RowRaybasedSampleDataset(indices, scene, params, out_w, out_h, rows_per_batch);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset, options);

        std::vector<torch::Tensor> projection_images(scene->frames.size());
        std::vector<torch::Tensor> target_images(scene->frames.size());
        std::vector<torch::Tensor> per_cell_loss_sum(scene->frames.size());



        for (int i : indices)
        {
            projection_images[i] = torch::zeros({1, out_h, out_w});
            target_images[i]     = torch::zeros({1, out_h, out_w});
            per_cell_loss_sum[i] = torch::zeros({(long)tree->NumNodes()}, device);
        }

        {
            // optimize network with fixed structure
            // Saiga::ProgressBar bar(std::cout,
            //                        name + " (" + std::to_string(out_w) + "x" + std::to_string(out_h) + ") " +
            //                            std::to_string(epoch_id) + " |",
            //                        out_w * out_h * indices.size(), 30, false, 1000, "ray");

            Saiga::ProgressBar bar(std::cout,
                                   name + " " + std::to_string(out_w)  + " " +
                                       std::to_string(epoch_id) + " |",
                                   out_w * out_h * indices.size(), 10, false, 1000, "ray");

            for (RowSampleData sample_data : (*data_loader))
            {
                RayList rays;
                {
                    rays =
                        scene->GetRays(sample_data.pixels.uv, sample_data.pixels.image_id, sample_data.pixels.camera_id);
                }


                SampleList all_samples;
                if(params->octree_params.tree_depth == 0)
                {
                    // printf("test here create samples for rays\n");
                    all_samples = tree->CreateSamplesForRays_area(rays, params->octree_params.max_samples_per_node,
                                                                    false,
                                                                    scene->dataset_params.roi_min,
                                                                    scene->dataset_params.roi_max);
                }
                else
                {
                    all_samples =
                        tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, false);
                }
                // SampleList all_samples =
                //     tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, false);

                auto    [predicted_image, ratio, ratio_inv] =
                    neural_geometry->ComputeImage(all_samples, scene->num_channels, sample_data.NumPixels(), params->net_params.using_decoder);
                (void) ratio_inv;
                predicted_image =
                    scene->tone_mapper->forward(predicted_image, sample_data.pixels.uv, sample_data.pixels.image_id);

                if (sample_data.pixels.target_mask.defined())
                {
                    // Multiply by mask so the loss of invalid pixels is 0
                    predicted_image           = predicted_image * sample_data.pixels.target_mask;
                    sample_data.pixels.target = sample_data.pixels.target * sample_data.pixels.target_mask;
                }

                CHECK_EQ(predicted_image.sizes(), sample_data.pixels.target.sizes());
                torch::Tensor per_ray_loss_mse;
                // printf("ratio here");
                // PrintTensorInfo(ratio);
                if(ratio.defined())
                {
                    per_ray_loss_mse = ((predicted_image - sample_data.pixels.target * ratio)).square();
                }
                else
                {
                    per_ray_loss_mse = ((predicted_image - sample_data.pixels.target )).square();
                }
                // new add constraint
                if(params->octree_params.use_estimate_update_tree)
                {
                    per_ray_loss_mse = predicted_image;
                }

                // CHECK_EQ(image_samples.image_index.size(0), 1);
                int image_id = sample_data.image_id;

                if (projection_images[image_id].is_cpu())
                {
                    for (auto& i : projection_images)
                    {
                        if (i.defined()) i = i.cpu();
                    }
                    projection_images[image_id] = projection_images[image_id].cuda();
                    target_images[image_id]     = target_images[image_id].cuda();
                }
                auto prediction_rows = predicted_image.reshape({predicted_image.size(0), -1, out_w});
                auto target_rows = sample_data.pixels.target.reshape({sample_data.pixels.target.size(0), -1, out_w});

                projection_images[image_id].slice(1, sample_data.row_start, sample_data.row_end) = prediction_rows;
                target_images[image_id].slice(1, sample_data.row_start, sample_data.row_end)     = target_rows;

                // std::cout << "image id " << image_id << TensorInfo(projection_images[image_id]) << std::endl;
                auto [loss_sum, weight_sum] =
                    neural_geometry->AccumulateSampleLossPerNode(all_samples, per_ray_loss_mse);
                (void) weight_sum;
                // PrintTensorInfo(loss_sum);
                // PrintTensorInfo(loss_sum2);

                per_cell_loss_sum[image_id] += loss_sum.detach();


                epoch_loss_train_mse += per_ray_loss_mse.mean().item().toFloat();


                image_count += sample_data.batch_size;

                // bar.SetPostfix(" MSE=" + std::to_string(epoch_loss_train_mse / image_count));
                bar.addProgress(sample_data.NumPixels());
            }
        }
        if (!test && params->octree_params.optimize_structure && calculate_std)
        {
            torch::Tensor cell_loss_combined;

            if (params->octree_params.using_std_error )
            {
                printf("update using std \n");
                cell_loss_combined = ComputeStd(ts);

            }
            else if (params->octree_params.mult_error)
            {
                cell_loss_combined = torch::ones({(long)tree->NumNodes()}, device);
                for (int i : indices)
                {
                    cell_loss_combined *= per_cell_loss_sum[i];
                }
            }

            else
            {
                printf("update use back propagation\n");
                cell_loss_combined = torch::zeros({(long)tree->NumNodes()}, device);
                for (int i : indices)
                {
                    cell_loss_combined += per_cell_loss_sum[i];
                }
            }

            torch::Tensor cell_loss_combined_bak = torch::zeros({(long)tree->NumNodes()}, device);
            for (int i : indices)
            {
                cell_loss_combined_bak += per_cell_loss_sum[i];
            }

            printf("cell loss is \n");
            PrintTensorInfo(cell_loss_combined);

            // if(.use_tree_roi)
            vec3 tree_roi_min = params->octree_params.tree_optimizer_params.tree_roi_min;
            vec3 tree_roi_max = params->octree_params.tree_optimizer_params.tree_roi_max;

            vec3 tree_tv_roi_min = params->octree_params.tree_optimizer_params.tree_tv_roi_min;
            vec3 tree_tv_roi_max = params->octree_params.tree_optimizer_params.tree_tv_roi_max;
            // Eigen::Vector<float, 3>  tree_roi_min;
            // Eigen::Vector<float, 3>  tree_roi_max;
            // for(int i = 0; i < 3; ++i)
            // {
            //     tree_roi_min(i) = roi_min[i];
            //     tree_roi_max(i) = roi_max[i];
            // }

            if(params->octree_params.push_roi_to_finest)
            {
                std::cout <<"push region of interest to finest " << std::endl;
            }
            cell_loss_combined=cell_loss_combined.to(torch::kCPU);
            if(params->octree_params.tree_optimizer_params.use_tree_roi)
            {

                auto node_min =  tree->node_position_min.to(torch::kCPU);
                auto node_max =  tree->node_position_max.to(torch::kCPU);
                auto node_mid =  node_min + node_max;
                PrintTensorInfo( node_min);

                auto active_nodes_id = tree->active_node_ids.to(torch::kCPU);
                // cell_loss_combined = cell_loss_combined.to(torch::kCPU);
                // printf("test here 1.1\n");
                // PrintTensorInfo( tree->active_node_ids);
                // printf("test here 1.2\n");

                float epsilon = 0;
                if(params->octree_params.use_quad_tree_rep)
                epsilon = 1e-6;
                for (int i = 0; i < tree->NumActiveNodes(); ++i)
                {
                    int node_id            = active_nodes_id.data_ptr<long>()[i];
                    // printf("global_coordinate is %d", node_id);
                    // std::cout << "global coord is " << node_id << " ";
                    // std::cout << node_min_ptr[node_id * 3 ]
                    // << node_min_ptr[node_id * 3 + 1]
                    // << node_min_ptr[node_id * 3 + 2] ;
                    // std::cout << " " << cell_loss_combined.to(torch::kCPU).data_ptr<float>()[node_id];
                    float* node_min_ptr = node_min.data_ptr<float>()+ node_id * node_min.stride(0);
                    float* node_max_ptr = node_max.data_ptr<float>()+ node_id * node_min.stride(0);
                    float* node_mid_ptr = node_mid.data_ptr<float>()+ node_id * node_min.stride(0);
                    if(params->octree_params.push_roi_to_finest)
                    {
                        // keep the cell in the region of interest the same
                        if( in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, tree_roi_min, tree_roi_max, epsilon) )
                        {
                            cell_loss_combined.data_ptr<float>()[node_id] = 1e6;
                        }
                        if( in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, tree_tv_roi_min, tree_tv_roi_max, epsilon) )
                        {
                            cell_loss_combined.data_ptr<float>()[node_id] = 1e9;
                        }
                    }
                    else
                    {
                        // only change the cell in the region of interest
                        if(!in_roi(node_min_ptr, node_max_ptr, node_mid_ptr, tree_roi_min, tree_roi_max, epsilon) )
                        {
                            cell_loss_combined.data_ptr<float>()[node_id] = 0;
                        }
                    }

                    // std::cout << " after  " << cell_loss_combined.to(torch::kCPU).data_ptr<float>()[node_id] <<std::endl;
                }
            }
            cell_loss_combined = cell_loss_combined.to(device);
            tree->SetErrorForActiveNodes(cell_loss_combined, "override");
        }

        torch::Tensor predicted_image_all,target_image_all;
        std::string ep_str, ep_dir;
        if(params->train_params.save_temp )
        {
            ep_str = Saiga::leadingZeroString(epoch_id, 4);
            ep_dir = experiment_dir + "ep" + ep_str + "/";
            std::filesystem::create_directory(ep_dir);
            predicted_image_all = torch::empty({0, projection_images[indices[0]].size(1), projection_images[indices[0]].size(2)});
            target_image_all = torch::empty({0, projection_images[indices[0]].size(1), projection_images[indices[0]].size(2)});
        }



        if (!projection_images.empty())
        {
            epoch_loss_train_mse = 0;
            for (int i : indices)
            {
                auto predicted_image = projection_images[i].cpu().unsqueeze(0);
                auto ground_truth    = target_images[i].cpu().unsqueeze(0);
                if(params->train_params.save_temp)
                {
                    predicted_image_all = torch::cat({predicted_image_all,projection_images[i].cpu()}, 0);
                    target_image_all    = torch::cat({target_image_all, target_images[i].cpu()}, 0);
                }


                CHECK_EQ(ground_truth.dim(), 4);
                CHECK_EQ(predicted_image.dim(), 4);

                auto image_loss_mse = ((predicted_image - ground_truth)).square();
                auto image_loss_l1  = ((predicted_image - ground_truth)).abs();


                epoch_loss_train_mse += image_loss_mse.mean().item().toFloat();
                epoch_loss_train_l1 += image_loss_l1.mean().item().toFloat();

                // printf("image loss l1\n");
                // PrintTensorInfo(image_loss_l1);

                if (!checkpoint_name.empty())
                {
                    auto err_col_tens =
                        ColorizeTensor(image_loss_l1.squeeze(0).mean(0) * 4, colorizeTurbo).unsqueeze(0);
                    CHECK_EQ(err_col_tens.dim(), 4);



                    // auto scale = 1.f / ground_truth.max();
                    // ground_truth *= scale;
                    // predicted_image *= scale;

                    auto predicted_colorized =
                        ColorizeTensor(predicted_image.squeeze(0).squeeze(0) * 8, colorizeTurbo).unsqueeze(0);
                    // LogImage(tblogger.get(), TensorToImage<ucvec3>(predicted_colorized),
                    //          checkpoint_name + "/Render" + name + "/render_x8", i);

                    predicted_colorized =
                        ColorizeTensor(predicted_image.squeeze(0).squeeze(0), colorizeTurbo).unsqueeze(0);
                    //                    LogImage(tblogger.get(), TensorToImage<ucvec3>(predicted_colorized),
                    //                             checkpoint_name + "/Render" + name + "/render", i);

                    auto gt_colorized = ColorizeTensor(ground_truth.squeeze(0).squeeze(0), colorizeTurbo).unsqueeze(0);

                    if (scene->num_channels == 1)
                    {
                        ground_truth    = ground_truth.repeat({1, 3, 1, 1});
                        predicted_image = predicted_image.repeat({1, 3, 1, 1});
                    }

                    auto stack    = torch::cat({gt_colorized, predicted_colorized, err_col_tens}, 0);
                    auto combined = ImageBatchToImageRow(stack);
                    if (indices.size() == 1)
                    {
                        LogImage(tblogger.get(), TensorToImage<ucvec3>(combined),
                                 checkpoint_name + "/Render" + name + "/gt_render_l1error", epoch_id);
                    }
                    else
                    {
                        LogImage(tblogger.get(), TensorToImage<ucvec3>(combined),
                                 checkpoint_name + "/Render" + name + "/gt_render_l1error", i);
                    }
                }
            }
        }
        if(params->train_params.save_temp )
        {

            if (scene->dataset_params.log_space_input)
            {
                SaveHDRImageTensor(predicted_image_all.unsqueeze(0), ep_dir + "/" + scene->scene_name + "_reproj.hdr");
                SaveHDRImageTensor(target_image_all.unsqueeze(0), ep_dir + "/" + scene->scene_name + "_original.hdr");
            }
            else 
            {
                float factor = std::log(scene->dataset_params.xray_max/scene->dataset_params.xray_min);
                predicted_image_all = torch::exp(-factor * predicted_image_all);
                target_image_all    = torch::exp(-factor * target_image_all);
                SaveHDRImageTensor(predicted_image_all.unsqueeze(0), ep_dir + "/" + scene->scene_name + "_reproj.hdr");
                SaveHDRImageTensor(target_image_all.unsqueeze(0), ep_dir + "/" + scene->scene_name + "_original.hdr");
            }

        }



        epoch_loss_train_mse /= indices.size();
        epoch_loss_train_l1 /= indices.size();
        float epoch_loss_train_psnr = 10 * std::log10(1. / epoch_loss_train_mse);
        if (epoch_id > 0)
        {
            CHECK(std::isfinite(epoch_loss_train_psnr));
            // tblogger->add_scalar("Loss" + name + "/mse", epoch_id, epoch_loss_train_mse);
            // tblogger->add_scalar("Loss" + name + "/l1", epoch_id, epoch_loss_train_l1);
            tblogger->add_scalar("Loss" + name + "/psnr", epoch_id, epoch_loss_train_psnr);
        }
        std::cout << ConsoleColor::GREEN << "> Loss " << name << " sample h " << out_h << " sample w " << out_w << " | MSE " << epoch_loss_train_mse << " L1 "
                  << epoch_loss_train_l1 << " PSNR " << epoch_loss_train_psnr << ConsoleColor::RESET << std::endl;
        // std::cout << ConsoleColor::GREEN << "out h " << out_h << "out_w " << out_w << ConsoleColor::RESET << std::endl;
        return epoch_loss_train_mse;
    }


    torch::Tensor Eval_cell_density(TrainScene& ts, std::vector<int> indices, std::string name, int epoch_id)
    {
        auto scene           = ts.scene;
        auto tree            = ts.tree;
        auto neural_geometry = ts.neural_geometry;
        // neural_geometry->train(epoch_id, false);
        // scene->train(false);
        // torch::NoGradGuard ngg;

        // double epoch_loss_train_l1  = 0;
        double epoch_loss_train_mse = 0;
        int image_count             = 0;

        // bool mult_error = false;


        int out_w = scene->cameras.front().w; //* params->train_params.eval_scale;
        int out_h = scene->cameras.front().h;// * params->train_params.eval_scale;

        printf("evaluate tv para setup %d %d\n", out_w, out_h);
        int rows_per_batch = std::max(params->train_params.batch_size / out_w, 1);


        auto options = torch::data::DataLoaderOptions().batch_size(1).drop_last(false).workers(
            params->train_params.num_workers_eval);
        auto dataset     = RowRaybasedSampleDataset(indices, scene, params, out_w, out_h, rows_per_batch);
        auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(dataset, options);

        // std::vector<torch::Tensor> projection_images(scene->frames.size());
        // std::vector<torch::Tensor> target_images(scene->frames.size());
        std::vector<torch::Tensor> per_cell_loss_sum(scene->frames.size());



        for (int i : indices)
        {
            // projection_images[i] = torch::zeros({1, out_h, out_w});
            // target_images[i]     = torch::zeros({1, out_h, out_w});
            per_cell_loss_sum[i] = torch::zeros({(long)tree->NumNodes()}, device);
        }

        {
            // optimize network with fixed structure
            Saiga::ProgressBar bar(std::cout,
                                   name + " (" + std::to_string(out_w) + "x" + std::to_string(out_h) + ") " +
                                       std::to_string(epoch_id) + " |",
                                   out_w * out_h * indices.size(), 10, false, 1000, "ray");

            for (RowSampleData sample_data : (*data_loader))
            {
                RayList rays =
                    scene->GetRays(sample_data.pixels.uv, sample_data.pixels.image_id, sample_data.pixels.camera_id);

                SampleList all_samples;
                if(params->octree_params.tree_depth == 0)
                {
                    all_samples = tree->CreateSamplesForRays_area(rays, params->octree_params.max_samples_per_node,
                                                                    params->train_params.train_interval_jitter,
                                                                    scene->dataset_params.roi_min,
                                                                    scene->dataset_params.roi_max);
                }
                else
                {
                    all_samples =
                        tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, false);
                }
                // SampleList all_samples =
                //     tree->CreateSamplesForRays(rays, params->octree_params.max_samples_per_node, false);

                // auto predicted_image =
                //     neural_geometry->ComputeImage(all_samples, scene->num_channels, sample_data.NumPixels());
                // predicted_image =
                //     scene->tone_mapper->forward(predicted_image, sample_data.pixels.uv, sample_data.pixels.image_id);

                if (sample_data.pixels.target_mask.defined())
                {
                    // Multiply by mask so the loss of invalid pixels is 0
                    // predicted_image           = predicted_image * sample_data.pixels.target_mask;
                    sample_data.pixels.target = sample_data.pixels.target * sample_data.pixels.target_mask;
                }

                // CHECK_EQ(predicted_image.sizes(), sample_data.pixels.target.sizes());
                auto per_ray_loss_mse = (( sample_data.pixels.target)).square();


                // CHECK_EQ(image_samples.image_index.size(0), 1);
                int image_id = sample_data.image_id;

                // if (projection_images[image_id].is_cpu())
                // {
                //     for (auto& i : projection_images)
                //     {
                //         if (i.defined()) i = i.cpu();
                //     }
                //     // projection_images[image_id] = projection_images[image_id].cuda();
                //     target_images[image_id]     = target_images[image_id].cuda();
                // }
                // auto prediction_rows = predicted_image.reshape({predicted_image.size(0), -1, out_w});
                // auto target_rows = sample_data.pixels.target.reshape({sample_data.pixels.target.size(0), -1, out_w});

                // projection_images[image_id].slice(1, sample_data.row_start, sample_data.row_end) = prediction_rows;
                // target_images[image_id].slice(1, sample_data.row_start, sample_data.row_end)     = target_rows;

                auto [loss_sum, weight_sum] =
                    neural_geometry->AccumulateSampleLossPerNode(all_samples, per_ray_loss_mse);
                (void) weight_sum;
                // PrintTensorInfo(loss_sum);
                // PrintTensorInfo(loss_sum2);

                per_cell_loss_sum[image_id] += loss_sum.detach();


                epoch_loss_train_mse += per_ray_loss_mse.mean().item().toFloat();


                image_count += sample_data.batch_size;

                // bar.SetPostfix(" MSE=" + std::to_string(epoch_loss_train_mse / image_count));
                bar.addProgress(sample_data.NumPixels());
            }
        }

        printf("use back propagation as tv regularier\n");
        // torch::Tensor cell_loss_combined = torch::zeros({(long)tree->NumNodes()}, device);
        torch::Tensor cell_loss_combined = 0.0001 * torch::ones({(long)tree->NumNodes()}, device);
        for (int i : indices)
        {
            cell_loss_combined += per_cell_loss_sum[i];
        }
        PrintTensorInfo(cell_loss_combined);
        return cell_loss_combined;
    }

    bool in_roi(float * node_min_ptr, float * node_max_ptr, float * node_mid_ptr, vec3 roi_min, vec3 roi_max, float epsilon)
    {
        float roi_min_y = roi_min[1] - epsilon;
        float roi_max_y = roi_max[1] + epsilon;

        float roi_min_x = roi_min[0] ;
        float roi_max_x = roi_max[0] ;
        float roi_min_z = roi_min[2] ;
        float roi_max_z = roi_max[2] ;
        if( ((node_min_ptr[0] > roi_min_x && node_min_ptr[0] < roi_max_x )||
            (node_max_ptr[0] > roi_min_x && node_max_ptr[0] < roi_max_x )||
            (node_mid_ptr[0] > roi_min_x && node_mid_ptr[0] < roi_max_x )) &&
            ((node_min_ptr[1] > roi_min_y && node_min_ptr[1] < roi_max_y) ||
            (node_max_ptr[1] > roi_min_y && node_max_ptr[1] < roi_max_y) ||
            (node_mid_ptr[ 1] > roi_min_y && node_mid_ptr[1] < roi_max_y)) &&
            ((node_min_ptr[2] > roi_min_z && node_min_ptr[2] < roi_max_z) ||
            (node_max_ptr[ 2] > roi_min_z && node_max_ptr[2] < roi_max_z) ||
            (node_mid_ptr[ 2] > roi_min_z && node_mid_ptr[2] < roi_max_z))  )
        {
            return true;
        }
        else
        {
            return false;
        }
    }
    void OptimizeTreeStructure(TrainScene& ts, int epoch_id)
    {
        {
            auto scene           = ts.scene;
            auto tree            = ts.tree;
            auto neural_geometry = ts.neural_geometry;
            torch::Tensor grid_interpolated;
            torch::Tensor old_mask;

            tree->to(torch::kCPU);


            HyperTreeStructureOptimizer opt(tree, params->octree_params.tree_optimizer_params);
            if (opt.OptimizeTree())
            {
                // If the structure has changed we need to recompute some stuff
                if (params->train_params.reset_optimizer_after_structure_change)
                {
                    std::cout << "Resetting Optimizer..." << std::endl;
                    neural_geometry->ResetGeometryOptimizer();
                    current_lr_factor = 1;
                }
            }
            tree->to(device);
            std::cout <<"after optimizing tree " << tree->NumActiveNodes() << std::endl;
        }
    }

    PSNR loss_function_psnr = PSNR(0, 1);

    std::shared_ptr<CombinedParams> params;
    std::string experiment_dir;

    std::vector<TrainScene> scenes;
    double current_lr_factor = 1;

   public:
    std::shared_ptr<TensorBoardLogger> tblogger;
};

template <typename ParamType>
ParamType LoadParamsHybrid(int argc, const char* argv[])
{
    CLI::App app{"Train The Hyper Tree", "hyper_train"};

    std::string config_file;
    app.add_option("config_file", config_file);
    auto params = ParamType();
    params.Load(app);
    app.parse(argc, argv);
    // CLI11_PARSE(app, argc, argv);
    std::cout << "Loading Config File " << config_file << std::endl;
    params.Load(config_file);

    // params.Load(app);
    app.parse(argc, argv);
    // CLI11_PARSE(app, argc, argv);


    // std::cout << app.help("", CLI::AppFormatMode::All) << std::endl;

    return params;
}




int main(int argc, const char* argv[])
{
    if (0)
    {
        for (int i = 0; i < 100; ++i)
        {
            float alpha = (i / (100.f - 1)) * 2 - 1;
            alpha *= 4;
            auto t  = torch::full({1}, alpha);
            auto t2 = torch::softplus(t, 2);
            auto t3 = torch::softplus(t, 4);
            std::cout << t.item().toFloat() << ": " << t2.item().toFloat() << " " << t3.item().toFloat() << std::endl;
        }
        return 0;
    }

    if (0)
    {


        torch::Tensor image = torch::arange(0, 5, torch::kFloat32).expand({1, 1, 5, 5}).to(device).requires_grad_();
        std::cout << image.cpu().squeeze(0).squeeze(0) << std::endl;

        // auto grid = torch::nn::functional::affine_grid(torch::tenso)
        // std::vector<float> data = {-1.2, 0, -1, 0, -0.5, 0, 0, 0, 0.5, 0, 1, 0, 1.2, 0};
        std::vector<float> data = {1, 0};
        torch::Tensor grid      = torch::from_blob(data.data(), {(long)data.size()})
                                 .clone()
                                 .to(device)
                                 .reshape({1, 1, -1, 2})
                                 .requires_grad_();
        std::cout << grid.squeeze(0).squeeze(0) << std::endl;

        auto opt = torch::nn::functional::GridSampleFuncOptions().align_corners(true).padding_mode(torch::kBorder);
        auto res = torch::nn::functional::grid_sample(image, grid, opt);
        PrintTensorInfo(res);
        res.sum().backward();

        std::cout << "grad" << std::endl;
        std::cout << grid.grad().cpu().squeeze(0).squeeze(0) << std::endl;
        std::cout << image.grad().cpu().squeeze(0).squeeze(0) << std::endl;

        return 0;
    }
    if (0)
    {
        // check max intensity
        std::vector<std::vector<double>> elemss;
        for (int image_id = 1; image_id <= 721; image_id += 2)
        {
            std::string n = leadingZeroString(image_id, 4);

            // TemplatedImage<unsigned short> img("/home/ruecked/datasets/Shared_Darius/Pepper/VCC_Fruit_0830_2 2_" + n
            // +
            //                                    ".tif");

            TemplatedImage<unsigned short> img("/home/ruecked/datasets/Shared_Darius/RopeBall/RopeBall_50kV_" + n +
                                               ".tif");
            // TemplatedImage<unsigned short> img_crop = img.getImageView().subImageView(20, 50, 1400 - 20, 1880 - 50);
            TemplatedImage<unsigned short> img_crop = img.getImageView().subImageView(20, 50, 1400 - 20, 1880 - 50);
            //            auto img_crop = img.getImageView().subImageView(20,50,100,100);

            std::cout << "crop " << img_crop << std::endl;
            std::vector<double> elems;
            ivec2 arg_max;
            double max_v = 0;

            for (int i : img_crop.rowRange())
            {
                for (int j : img_crop.colRange())
                {
                    double v = img_crop(i, j);
                    //                    double v = img(i, j);

                    if (v > max_v)
                    {
                        max_v   = v;
                        arg_max = ivec2(j, i);
                    }

                    elems.push_back(v);
                }
            }
            std::cout << "argmax: " << arg_max.transpose() << " " << max_v << std::endl;
            elemss.push_back(elems);
        }
        double total_max = 0;
        for (int i = 0; i < elemss.size(); ++i)
        {
            auto stat = Statistics(elemss[i]);
            std::cout << i << "\n" << stat << std::endl;
            total_max = std::max(stat.max, total_max);
        }
        std::cout << "Total max: " << total_max << std::endl;
        return 0;
    }

    auto params = std::make_shared<CombinedParams>(LoadParamsHybrid<CombinedParams>(argc, argv));

    // auto params = std::make_shared<CombinedParams>(LoadParamsHybrid(argc, argv));

    // std::string experiment_dir = PROJECT_DIR.append("Experiments");
    std::string experiment_dir = PROJECT_DIR.append(params->train_params.exp_dir_name);
    std::filesystem::create_directories(experiment_dir);
    experiment_dir = experiment_dir + "/" + params->train_params.ExperimentString() + "/";
    std::filesystem::create_directories(experiment_dir);
    params->Save(experiment_dir + "/params.ini");


    Trainer trainer(params, experiment_dir);

    std::string args_combined;
    for (int i = 0; i < argc; ++i)
    {
        args_combined += std::string(argv[i]) + " ";
    }
    trainer.tblogger->add_text("arguments", 0, args_combined.c_str());
    trainer.Train();


    return 0;
}
