/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#pragma once

#include "saiga/core/image/freeimage.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ImageTensor.h"
#include "saiga/vision/torch/TrainParameters.h"

#include "modules/ToneMapper.h"
#include "structure/HyperTreeStructureOptimizer.h"

#include "tensorboard_logger.h"
using namespace Saiga;



#include "build_config.h"


struct Netparams : public ParamsBase
{
    // SAIGA_PARAM_STRUCT(Netparams);
    // SAIGA_PARAM_STRUCT_FUNCTIONS;

    // template <class ParamIterator>
    // void Params(ParamIterator* it) 
    SAIGA_PARAM_STRUCT_FUNCTIONS(Netparams);
    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM(grid_size);
        SAIGA_PARAM(rand_grid_size); // new add
        SAIGA_PARAM(edge_grid_size); // new add
	    SAIGA_PARAM(NLM_grid_size);	// new add
        SAIGA_PARAM(std_grid_size); // new add
        SAIGA_PARAM(edge_grid_scale); // new add
        SAIGA_PARAM_LIST(vec_grid_feature, ' '); // new add
        SAIGA_PARAM_LIST(vec_grid_size, ' '); // new add

        SAIGA_PARAM(intermidiate_features); // new add


        SAIGA_PARAM(grid_features);
        SAIGA_PARAM(hidden_features);
        SAIGA_PARAM(hidden_layers);
        SAIGA_PARAM(geometry_type);
        SAIGA_PARAM(num_encoding_functions);
        SAIGA_PARAM(sampler_align_corners);
        SAIGA_PARAM(last_activation_function);
        SAIGA_PARAM(softplus_beta);

        SAIGA_PARAM(gaussian_pe_num_ef);    // new add
        SAIGA_PARAM(gaussian_pe_std);       // new add

        SAIGA_PARAM(kilo_activation);
        SAIGA_PARAM(kilo_hidden_layers);
        SAIGA_PARAM(kilo_hidden_features);
        SAIGA_PARAM(kilo_encoding_functions);

        SAIGA_PARAM(decoder_skip);
        SAIGA_PARAM(generator_activation);
        SAIGA_PARAM(decoder_lr);
        SAIGA_PARAM(decoder_activation);
        SAIGA_PARAM(decoder_hidden_layers);
        SAIGA_PARAM(decoder_hidden_features);

        SAIGA_PARAM(conv_features);         // new add
        SAIGA_PARAM(conv_lr);               // new add


        // Fourier Feature Net
        SAIGA_PARAM(using_fourier);            // new add Fourier feature net
        SAIGA_PARAM(adding_fourier_weight);    // new add
        // SAIGA_PARAM(fourier_hidden_features);
        SAIGA_PARAM(fourier_num_hidden_layers);
        SAIGA_PARAM_LIST(fourier_frequency,' ');
        SAIGA_PARAM_LIST(fourier_input_scale,' ');
        SAIGA_PARAM(update_tv_its);     // new add

        SAIGA_PARAM(using_reverse_tv);  // new add
        SAIGA_PARAM(using_decoder);  // new add
        SAIGA_PARAM(using_tensor_decoder); // new add
        SAIGA_PARAM(decoder_lr_decay);      // new add
        SAIGA_PARAM(use_tree_line);         // new add

        SAIGA_PARAM(hash_num_levels);
        SAIGA_PARAM(hash_level_dim);
        SAIGA_PARAM(hash_per_level_scale);
        SAIGA_PARAM(hash_base_resolution);
        SAIGA_PARAM(hash_log2_hashsize);
        SAIGA_PARAM(hash_desired_resolution);

    }

    int hash_num_levels         = 16;
    int hash_level_dim          = 2;
    int hash_per_level_scale    = 2;
    int hash_base_resolution    = 16;
    int hash_log2_hashsize      = 19;
    int hash_desired_resolution = 4096;
    int std_grid_size           = 20;

    bool using_decoder          = true;
    bool using_tensor_decoder   = false;

    int update_tv_its                   = 4;
    bool using_reverse_tv       = false;
    int conv_features           = 16; // new add

    int edge_grid_size          = 16;
    int rand_grid_size          = 16; // new add random sampling grid size
    float edge_grid_scale       = 0.8;


    int NLM_grid_size 		= 15; // non local grid size

    double conv_lr              = 0.0001; // new add

    int gaussian_pe_num_ef      = 256;
    float gaussian_pe_std       = 20;

    // Fourier Feature Net
    bool using_fourier              = false; // new add
    bool adding_fourier_weight      = false; // new add
    // int fourier_hidden_features     = 256;  // new add
    int fourier_num_hidden_layers   = 2;
    vec3 fourier_frequency          = vec3(16.,16.,16.);
    vec3 fourier_input_scale        ;


    int intermidiate_features   = 12;
    int grid_size               = 17;
    int grid_features           = 8;
    int hidden_features         = 512;
    int hidden_layers           = 4;
    int num_encoding_functions  = 6;
    vec3 vec_grid_feature       = vec3(10, 20, 10) ;
    vec3 vec_grid_size          = vec3(20, 100, 20);

    // acorn, kilo, doublekilo
    std::string geometry_type = "exex";

    // relu, id, abs
    std::string last_activation_function = "softplus";
    float softplus_beta                  = 2;

    bool sampler_align_corners = true;


    std::string kilo_activation = "silu";
    int kilo_hidden_layers      = 1;
    int kilo_hidden_features    = 32;
    int kilo_encoding_functions = 6;

    std::string generator_activation = "silu";
    bool decoder_skip                = false;
    float decoder_lr                 = 0.0005;
    std::string decoder_activation   = "silu";
    int decoder_hidden_layers        = 1;
    int decoder_hidden_features      = 64;

    float decoder_lr_decay           = 0.1; // new add
    bool use_tree_line               = true; // new add
};

// Params for the HyperTree
struct OctreeParams : public ParamsBase
{
    // old style
    SAIGA_PARAM_STRUCT_FUNCTIONS(OctreeParams);
    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override

    // SAIGA_PARAM_STRUCT(OctreeParams);
    // SAIGA_PARAM_STRUCT_FUNCTIONS;
    // template <class ParamIterator>
    // void Params(ParamIterator* it) 
    {
        SAIGA_PARAM(start_layer);
        SAIGA_PARAM(tree_depth);
        SAIGA_PARAM(optimize_structure);

        SAIGA_PARAM(use_quad_tree_rep); // use quad tree to represent three d scene

        SAIGA_PARAM(optimize_structure_last);
        SAIGA_PARAM(max_samples_per_node);
        SAIGA_PARAM(culling_start_epoch);
        SAIGA_PARAM(node_culling);
        SAIGA_PARAM(culling_threshold);
        SAIGA_PARAM(cull_at_epoch_zero);

        SAIGA_PARAM(mult_error); // new add
        SAIGA_PARAM(using_std_error); // new add
        SAIGA_PARAM(use_estimate_update_tree); //new add
        SAIGA_PARAM(tree_optimizer_params.use_tree_roi); // new add
        SAIGA_PARAM(tree_optimizer_params.optimize_tree_roi_at_ini); // new add
        SAIGA_PARAM_LIST(tree_optimizer_params.tree_roi_min, ' '); //new add
        SAIGA_PARAM_LIST(tree_optimizer_params.tree_roi_max, ' '); //new add
        SAIGA_PARAM(tree_optimizer_params.use_tree_roi_nb); // new add

        SAIGA_PARAM_LIST(tree_optimizer_params.tree_ini_roi_min, ' '); //new add
        SAIGA_PARAM_LIST(tree_optimizer_params.tree_ini_roi_max, ' '); //new add

        SAIGA_PARAM_LIST(tree_optimizer_params.tree_edge_roi_min,' '); // new add
        SAIGA_PARAM_LIST(tree_optimizer_params.tree_edge_roi_max,' '); // new add

        // SAIGA_PARAM_LIST(tree_optimizer_params.tree_tv_roi_min,' '); // new add
        // SAIGA_PARAM_LIST(tree_optimizer_params.tree_tv_roi_max,' '); // new add
        // SAIGA_PARAM_LIST(tree_optimizer_params.nlm_roi_min,' '); // new add
        // SAIGA_PARAM_LIST(tree_optimizer_params.nlm_roi_max,' '); // new add
        SAIGA_PARAM(std_iteration); //new add

        SAIGA_PARAM(push_roi_to_finest); // new add

        SAIGA_PARAM(tree_optimizer_params.use_saved_errors);
        SAIGA_PARAM(tree_optimizer_params.max_active_nodes);
        SAIGA_PARAM(tree_optimizer_params.max_active_nodes_initial);
        SAIGA_PARAM(tree_optimizer_params.error_merge_factor);
        SAIGA_PARAM(tree_optimizer_params.error_split_factor);
        SAIGA_PARAM(tree_optimizer_params.verbose);
    }

    bool mult_error                     = false; // new add using multiply error as tree update
    bool using_std_error                = true;  // new add using std error to update tree
    int  std_iteration                  = 100;
    bool use_tree_roi_nb                = false;

    bool use_quad_tree_rep              = true;

    bool optimize_structure_last        = false; // new add update the octree update at last octree level

    bool push_roi_to_finest             = false;    // new add push the roi into finest
    bool optimize_tree_roi_at_ini       = false;

    bool use_tree_roi = true; //new add
    vec3 tree_roi_min = vec3(-1, -1, -1); //new add
    vec3 tree_roi_max = vec3(1, 1, 1); //new add
    bool use_estimate_update_tree = false; //new add
    vec3 tree_ini_roi_min = vec3(-1, -1, -1); //new add
    vec3 tree_ini_roi_max = vec3(1, 1, 1); //new add

    vec3 tree_edge_roi_min = vec3(-1, -1, -1); // new add
    vec3 tree_edge_roi_max = vec3(1, 1, 1); // new add

    // vec3 tree_tv_roi_min        = vec3(-1, -1, -1); // new add
    // vec3 tree_tv_roi_max        = vec3(1,1,1); // new add
    // vec3 nlm_roi_min        = vec3(-1, -1, -1); // new add
    // vec3 nlm_roi_max        = vec3(-1, -1, -1); // new add


    int start_layer         = 3;
    int tree_depth          = 4;
    bool optimize_structure = true;

    int max_samples_per_node = 32;

    // can be used to cull away all invisible nodes
    bool cull_at_epoch_zero = false;

    int culling_start_epoch = 5;
    bool node_culling       = true;

    // 0.01 for mean, 0.4 for max
    float culling_threshold = 0.2;


    TreeOptimizerParams tree_optimizer_params;
};

struct DatasetParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(DatasetParams);
    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    // SAIGA_PARAM_STRUCT(DatasetParams);
    // SAIGA_PARAM_STRUCT_FUNCTIONS;
    // template <class ParamIterator>
    // void Params(ParamIterator* it) 
    {
        SAIGA_PARAM(camera_model);
        SAIGA_PARAM(image_dir);
        SAIGA_PARAM(mask_dir);
        SAIGA_PARAM(projection_factor);
        SAIGA_PARAM(vis_volume_intensity_factor);
        SAIGA_PARAM(scene_scale);
        SAIGA_PARAM(xray_min);
        SAIGA_PARAM(xray_max);
        SAIGA_PARAM(volume_file);
        SAIGA_PARAM(log_space_input);
        SAIGA_PARAM(use_log10_conversion);

        SAIGA_PARAM(projection_scale); //new add
        SAIGA_PARAM(laplacian_size) ; //new add
        SAIGA_PARAM(camera_proj_scale); //new add
        SAIGA_PARAM(slice_min); // new add
        SAIGA_PARAM(slice_max); // new add

        SAIGA_PARAM_LIST(roi_min, ' ');
        SAIGA_PARAM_LIST(roi_max, ' ');
    }

    float camera_proj_scale = 1.875;
    int laplacian_size = 5;
    int projection_scale = 3; //new add
    // supported: "pinhole", "orthographic"
    std::string camera_model = "pinhole";

    // only set if a ground truth volume exists
    std::string volume_file = "";

    int slice_min   = 125;
    int slice_max   = 375;

    // linear multiplier to the projection
    // otherwise it is just transformed by xray/min/max parameters
    double projection_factor = 1;


    // Only for visualization!
    // multiplied to the intensity of the projection (after normalization)
    double vis_volume_intensity_factor = 1;

    // the camera position is multiplied by this factor to "scale" the scene
    double scene_scale = 1;

    std::string image_dir = "";
    std::string mask_dir  = "";

    // "real" raw xray is usually NOT in log space (background is white)
    // if the data is already preprocessed and converted to log space (background is black)
    // set this flag in the dataset
    bool log_space_input = false;

    // pepper:13046, 65535
    // ropeball: 26000, 63600
    double xray_min = 0;
    double xray_max = 65535;

    // By default we reconstruct the unit cube [-1^3 , 1^3].
    // Changing these values culls (set to zero) all nodes outside of that region
    // Nodes that are partly inside will not be culled
    vec3 roi_min = vec3(-1, -1, -1);
    vec3 roi_max = vec3(1, 1, 1);

    // true: log10
    // false: loge
    bool use_log10_conversion = true;
};

struct MyTrainParams : public TrainParams
{
    // using ParamStructType = MyTrainParams;

    // MyTrainParams(){}
    // MyTrainParams(const std::string file) : TrainParams(file) {}

    // SAIGA_PARAM_STRUCT_FUNCTIONS;
    // template <class ParamIterator>
    // void Params(ParamIterator* it) 

    MyTrainParams() {}
    MyTrainParams(const std::string file) { Load(file); }

    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        // TrainParams::Params(it);

        TrainParams::Params(ini, app);

        // input/output setting
        SAIGA_PARAM(scene_dir);
        SAIGA_PARAM_LIST2(scene_name, ' ');
        SAIGA_PARAM(split_name);
        SAIGA_PARAM(exp_dir_name);
        SAIGA_PARAM(volume_slice_num); // new add
        SAIGA_PARAM(volume_slice_view); // new add
        SAIGA_PARAM(rays_per_image);
        SAIGA_PARAM(output_volume_size);
        SAIGA_PARAM(output_volume_scale); // new add
        SAIGA_PARAM(output_dim);
        SAIGA_PARAM(out_roi_scale); // new add 
        SAIGA_PARAM(use_ground_truth_volume); // new add
        SAIGA_PARAM(save_temp);

        // basic training setting
        SAIGA_PARAM(per_node_batch_size);
        SAIGA_PARAM(per_node_vol_batch_size); // new add
        SAIGA_PARAM(train_interval_jitter);
        SAIGA_PARAM(interpolate_samples);
        SAIGA_PARAM(block_random);
        SAIGA_PARAM_LIST(volume_translation, ' '); // new add
        SAIGA_PARAM(use_half_precision);
        SAIGA_PARAM(eval_scale);

        // tree structure optimization setting
        SAIGA_PARAM(optimize_structure_every_epochs);
        SAIGA_PARAM(optimize_structure_convergence);
        SAIGA_PARAM(reset_optimizer_after_structure_change);
        SAIGA_PARAM(current_scale); //new add
        SAIGA_PARAM(scale_0_iter); // new add
        SAIGA_PARAM(optimize_structure_after_epochs);
        SAIGA_PARAM(optimize_tree_structure_after_epochs);
        SAIGA_PARAM(optimize_tone_mapper_after_epochs);
        // optimize the pose (position + rotation) and the exposure time
        // of the test frames.
        SAIGA_PARAM(optimize_test_image_params);
        

        // learning rate setting
        SAIGA_PARAM(lr_acorn);
        SAIGA_PARAM(exex_op);
        SAIGA_PARAM(lr_exex_grid_rms);
        SAIGA_PARAM(lr_exex_grid_adam);
        SAIGA_PARAM(lr_double_kilo);
        SAIGA_PARAM(lr_nerf);
        SAIGA_PARAM(kilo_lr);
        SAIGA_PARAM(optimize_pose);
        SAIGA_PARAM(optimize_intrinsics);
        SAIGA_PARAM(lr_pose);
        SAIGA_PARAM(lr_intrinsics);
        SAIGA_PARAM(lr_decay_factor);
        SAIGA_PARAM(init_bias_with_bg);
        SAIGA_PARAM(grid_init);
        SAIGA_PARAM(gradient_correction);

        // loss setting
        SAIGA_PARAM(plane_vec_only);        // new add
        SAIGA_PARAM(plane_op);              // new add
        SAIGA_PARAM(tensor_op); // new add
        SAIGA_PARAM(loss_scale);
        SAIGA_PARAM(loss_weighted);
        SAIGA_PARAM(loss_mse_scale);
        SAIGA_PARAM(loss_l1_scale);
        SAIGA_PARAM(loss_zero);
        SAIGA_PARAM(num_zero_epochs);
        SAIGA_PARAM(tv_loss_in_feature_space);
        SAIGA_PARAM(loss_tv);
        SAIGA_PARAM(huber_loss_factor);     // new add
        SAIGA_PARAM(loss_sparse);           // new add 
        SAIGA_PARAM(loss_edge);
        SAIGA_PARAM(loss_l2);
        SAIGA_PARAM(struct_opt_volume_size);
        SAIGA_PARAM(test_line_x);       // to remove
        SAIGA_PARAM(test_line_y);       // to remove
        SAIGA_PARAM(test_line_z);       // to remove
        SAIGA_PARAM(vm_init_scale);     // to remove
        SAIGA_PARAM(use_global_tv);     // new add
        SAIGA_PARAM(loss_tv_global);    // new add
        SAIGA_PARAM(use_tv_roi);        // new add
        SAIGA_PARAM(exex_using_pe); // new add
        SAIGA_PARAM(use_loss_tv_3d);        // new add
        SAIGA_PARAM(loss_tv_scale);     // new add
        SAIGA_PARAM(use_cross_nlm);
        SAIGA_PARAM(use_weighted_nlm);
        SAIGA_PARAM(use_NLM);		// new add
        SAIGA_PARAM(nlm_stop);
        SAIGA_PARAM(loss_nlm);      // new add
        SAIGA_PARAM(loss_nlm_scale);      // new add
        SAIGA_PARAM(nlm_w);
        SAIGA_PARAM(nlm_h); 
        SAIGA_PARAM(use_loss_nlm_vgg);      // new add
        SAIGA_PARAM(loss_nlm_vgg);          // new add
        SAIGA_PARAM(num_features_vgg);      // new add
        SAIGA_PARAM(nlm_vgg_scale);         // new add
        SAIGA_PARAM(vgg_net_path);          // new add
        SAIGA_PARAM(vgg_loss_type);         // new add
        SAIGA_PARAM(use_nlm_loss_density);      // new add
        SAIGA_PARAM(nlm_loss_density_num);      // new add
        SAIGA_PARAM_LIST(nlm_roi,' ');                   // new add
        SAIGA_PARAM(use_line_nlm);              // new add
        SAIGA_PARAM(loss_fourier);          // new add 
        SAIGA_PARAM(loss_fourier_scale);    // new add 
        SAIGA_PARAM(loss_fourier_scale_y);  // new add
        SAIGA_PARAM(loss_fourier_shift);    // new add 
        SAIGA_PARAM(loss_fourier_rate);
        SAIGA_PARAM(loss_fourier_rate_w);
        SAIGA_PARAM(use_vec_feature);       // to remove
        SAIGA_PARAM(test_para);

        SAIGA_PARAM(noise_translation);
        SAIGA_PARAM(noise_rotation);
        SAIGA_PARAM(noise_intrinsics);
        SAIGA_PARAM(use_moment);            // new add
        SAIGA_PARAM(moment_start_epochs); // new add
        SAIGA_PARAM(update_grid_each_iter); // new add

        // defocus setting
        SAIGA_PARAM(use_defocus);           // new add
        SAIGA_PARAM(defocus_std_init);      // new add
        SAIGA_PARAM(defocus_max_blur);      // new add
        

        SAIGA_PARAM(noise_level);
    }

    std::string                         scene_dir = "";
    std::vector<std::string>           scene_name = {"pepper"};
    std::string                        split_name = "exp_uniform_50";
    std::string                      exp_dir_name = "Experiments";
    int                          volume_slice_num = 64;
    int                         volume_slice_view = 200;
    int                            rays_per_image = 500000;
    int                        output_volume_size = 64;
    int                       output_volume_scale = 2;
    int                                output_dim = 0; //z
    float                           out_roi_scale = 0.5; // new add
    bool                  use_ground_truth_volume = false;
    bool                                save_temp = false;

    // basic training setting
    int                       per_node_batch_size = 256;
    int                   per_node_vol_batch_size = 64;  // per_node_vol_batch_size should be divided by the volume size
    bool                    train_interval_jitter = true;
    bool                      interpolate_samples = false;
    bool                             block_random = true;
    vec3                       volume_translation = vec3(0., 0., 0.); //new add
    bool                       use_half_precision = true;
    double                             eval_scale = 0.25;

    // tree structure optimization setting
    int           optimize_structure_every_epochs = 5;
    float          optimize_structure_convergence = 0.95;
    bool   reset_optimizer_after_structure_change = true;
    int                             current_scale = 1;    // new add current pramid scale
    int                              scale_0_iter = 21;   // last iteration for larger volume
    int           optimize_structure_after_epochs = 5;
    int      optimize_tree_structure_after_epochs = 0;
    int         optimize_tone_mapper_after_epochs = 1;
    bool               optimize_test_image_params = false;

    // learning rate setting
    float                                lr_acorn = 0.001;
    std::string                           exex_op = "rms";
    float                        lr_exex_grid_rms = 0.01;
    float                       lr_exex_grid_adam = 0.01;
    float                          lr_double_kilo = 0.00002;
    float                                 lr_nerf = 0.0005;
    float                                 kilo_lr = 0.004;
    bool                            optimize_pose = true;
    bool                      optimize_intrinsics = true;
    float                                 lr_pose = 0.001;
    float                           lr_intrinsics = 100;
    double                        lr_decay_factor = 0.95;
    // On each image we compute the median value of a top right corner crop
    // This is used to initialize the tone-mapper's bias value
    bool                        init_bias_with_bg = true;
    std::string                         grid_init = "uniform";
    bool                      gradient_correction = true;

    // loss setting
    bool plane_vec_only                 = true;  // new add
    std::string plane_op                = "add"; // new add add multi cat
    std::string tensor_op               = "add"; // new add
    double loss_scale        = 1;
    double loss_weighted     = -1;
    double loss_mse_scale    = 1;
    double loss_l1_scale     = 0;
    double loss_zero              = 0.01;
    int num_zero_epochs           = 2;  // disable zero loss after this amount of epochs
    // if true we compute the tv loss on the neural features
    // otherwise tv is computed on the final decoded density
    bool tv_loss_in_feature_space = true;
    double loss_tv                = 0.0005;
    float huber_loss_factor     = 0.5f;
    float loss_sparse                   = 0.0001;
    double loss_edge              = 0.002;
    float loss_l2                       = -0.001;   // new add
    int struct_opt_volume_size = 0;
    float test_line_x                   = 1; // to remove
    float test_line_y                   = 1; // to remove
    float test_line_z                   = 1; // to remove
    float vm_init_scale                 = 0.5; // to remove
    bool use_global_tv                  = true; // new add use global tv
    float loss_tv_global                = 0.002; // global tv loss
    bool use_tv_roi                     = false; // new add
    bool exex_using_pe          = true;
    bool use_loss_tv_3d                     = true; // new add
    float loss_tv_scale                 = 0.8;  // new add
    bool use_cross_nlm          = false; // use cross nlm
    bool use_weighted_nlm       = false; // use weighted nlm
    bool use_NLM		        = false; // new add
    int nlm_stop                        = 10; // to remove
    float loss_nlm              = -0.0001; // new add
    float loss_nlm_scale        = 100.0;    // scale the weighted nlm loss
    float nlm_w                         = 8; // nlm
    float nlm_h                         = 0.05; // nlm standard deviation
    bool use_loss_nlm_vgg       = false; // new add
    float loss_nlm_vgg          = -0.0001; // new add
    int num_features_vgg        = 1000; // new add
    float nlm_vgg_scale         = 1.0f; // new add
    // bool reset_params_vgg       = false; // new add
    std::string vgg_net_path    = " "; // new add
    std::string vgg_loss_type   = "perceptual "; // new add
    bool use_nlm_loss_density   = false;
    int nlm_loss_density_num    = 10;
    vec2    nlm_roi             = vec2(-0.22,-0.1);
    bool use_line_nlm           = false;
    float loss_fourier                  = -0.0001; // 
    float loss_fourier_scale            = 1.0f;
    float loss_fourier_scale_y          = 1.0f;
    float loss_fourier_shift            = 0.05f;
    float loss_fourier_rate             = 0.2f;
    float loss_fourier_rate_w             = 0.01f;
    bool use_vec_feature                = false;
    float test_para                     = -1;
    double noise_translation = 0;
    double noise_rotation    = 0;
    double noise_intrinsics  = 0;
    bool use_moment                     = false;    // new add
    int moment_start_epochs             = 5;        // new add
    bool update_grid_each_iter  = false; // new add


    // defocus setting
    bool use_defocus            = false;
    float defocus_std_init      = 1.0f;
    float defocus_max_blur      = 1.0f;


    double noise_level                  = 0.05;
};



struct CombinedParams
{
    MyTrainParams train_params;
    OctreeParams octree_params;
    Netparams net_params;
    PhotometricCalibrationParams photo_calib_params;

    CombinedParams() {}
    CombinedParams(const std::string& combined_file)
        : train_params(combined_file),
          octree_params(combined_file),
          net_params(combined_file),
          photo_calib_params(combined_file)
    {
    }

    void Save(const std::string file)
    {
        train_params.Save(file);
        octree_params.Save(file);
        net_params.Save(file);
        photo_calib_params.Save(file);
    }

    void Load(std::string file)
    {
        train_params.Load(file);
        octree_params.Load(file);
        net_params.Load(file);
        photo_calib_params.Load(file);
    }

    void Load(CLI::App& app)
    {
        train_params.Load(app);
        octree_params.Load(app);
        net_params.Load(app);
        photo_calib_params.Load(app);
    }
};
