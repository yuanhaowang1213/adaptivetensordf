/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "SceneBase.h"

#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ColorizeTensor.h"

#include "utils/cimg_wrapper.h"
vec2 PixelToUV(int x, int y, int w, int h)
{
    CHECK_GE(x, 0);
    CHECK_GE(y, 0);
    ivec2 shape(h, w);
    CHECK_LT(x, shape(1));
    CHECK_LT(y, shape(0));
    // align corner = false
    // vec2 s  = vec2(shape(1), shape(0));
    // vec2 uv = (vec2(x, y) + vec2(0.5, 0.5)).array() / s.array();

    // align corner = true
    vec2 s  = vec2(shape(1) - 1, shape(0) - 1);
    vec2 uv = (vec2(x, y)).array() / s.array();
    // std::swap(uv(0), uv(1));
    return uv;
}

vec2 UVToPix(vec2 uv, int w, int h)
{
    ivec2 shape(h, w);
    vec2 s   = vec2(shape(1) - 1, shape(0) - 1);
    vec2 pix = (uv).array() * s.array();
    return pix;
}

torch::Tensor UnifiedImage::CoordinatesRandomNoInterpolate(int count, int w, int h)
{
    auto y = torch::randint(0, h, {count, 1}).to(torch::kInt).to(torch::kFloat);
    auto x = torch::randint(0, w, {count, 1}).to(torch::kInt).to(torch::kFloat);

    torch::Tensor coords = torch::cat({x, y}, 1);
    return PixelToUV(coords, w, h, uv_align_corners);
}

std::pair<torch::Tensor, torch::Tensor> UnifiedImage::CoordinatesRandomNoInterpolate(int count, int w, int h, torch::Tensor defocus)
{
    auto y = torch::randint(0, h, {count, 1}).to(torch::kInt).to(torch::kFloat);
    auto x = torch::randint(0, w, {count, 1}).to(torch::kInt).to(torch::kFloat);
    auto defocus_out = at::index_select(defocus, 0, x.squeeze(1).to(torch::kInt) );
    torch::Tensor coords = torch::cat({x, y}, 1);
    return {PixelToUV(coords, w, h, uv_align_corners), defocus_out};
}

torch::Tensor UnifiedImage::CoordinatesRow(int row_start, int row_end, int w, int h)
{
    int num_rows = row_end - row_start;
    CHECK_GT(num_rows, 0);
    CHECK_LE(row_end, h);

    torch::Tensor px_coords = torch::empty({w * num_rows, 2});
    vec2* pxs               = px_coords.data_ptr<vec2>();

    for (int row_id = row_start; row_id < row_end; ++row_id)
    {
        for (int x = 0; x < w; ++x)
        {
            pxs[(row_id - row_start) * w + x] = vec2(x, row_id);
        }
    }

    return PixelToUV(px_coords, w, h, uv_align_corners);
}
std::pair<torch::Tensor, torch::Tensor> UnifiedImage::SampleProjection(torch::Tensor uv)
{
    torch::nn::functional::GridSampleFuncOptions opt;
    opt.align_corners(uv_align_corners).padding_mode(torch::kBorder).mode(torch::kBilinear);
    uv = uv * 2 - 1;
    uv = uv.unsqueeze(0).unsqueeze(0);
    //        PrintTensorInfo(uv);
    //        PrintTensorInfo(projection);

    // [1, num_channels, 1, num_coords]
    auto samples = torch::nn::functional::grid_sample(projection.unsqueeze(0), uv, opt);
    samples      = samples.reshape({NumChannels(), -1});

    torch::Tensor samples_mask;
    if (mask.defined())
    {
        samples_mask = torch::nn::functional::grid_sample(mask.unsqueeze(0), uv, opt);
        samples_mask = samples_mask.reshape({1, -1});
    }

    return {samples, samples_mask};
}

void SceneBase::save(std::string dir) {}
void SceneBase::Finalize()
{
    std::vector<SE3> poses;
    for (auto& f : frames)
    {
        // poses.push_back(f->pose);
        auto transformed_pose = f->pose;
        // // printf("test here 0\n");
        // transformed_pose.translation() += -params->train_params.volume_translation.cast<double>();
        transformed_pose.translation() = transformed_pose.translation()-params->train_params.volume_translation.cast<double>();
        // std::cout << "try translate " << std::endl;
        // std::cout << params->train_params.volume_translation.cast<double>() << std::endl;
        // // printf("test here 1\n");
        poses.push_back(transformed_pose);
    }

    for(int i = 0; i < 3; ++i)
    vol_translate_para.data_ptr<float>()[i * vol_translate_para.stride(0)] = params->train_params.volume_translation(i);
    vol_translate_para =  vol_translate_para.to(device);
    std::vector<IntrinsicsPinholed> intrinsics;
    for (auto c : cameras)
    {
        intrinsics.push_back(c.K);
    }
    pose         = CameraPoseModule(poses);
    camera_model = CameraModelModule(cameras.front().h, cameras.front().w, intrinsics);

    // new add put record on original image camera level
    ori_w = cameras.front().w;
    ori_h = cameras.front().h;
    rays_per_image_level0 = params->train_params.rays_per_image;


    pose->to(device);
    camera_model->to(device);

    CHECK(params);
    CHECK(camera_model);
    tone_mapper = PhotometricCalibration(frames.size(), cameras.size(), camera_model->h, camera_model->w,
                                         params->photo_calib_params);
    tone_mapper->to(device);

    {
        std::vector<torch::optim::OptimizerParamGroup> g;
        if (params->train_params.optimize_pose)
        {
            g.emplace_back(pose->parameters(),
                           std::make_unique<torch::optim::SGDOptions>(params->train_params.lr_pose));
        }
        if (params->train_params.optimize_intrinsics)
        {
            g.emplace_back(camera_model->parameters(),
                           std::make_unique<torch::optim::SGDOptions>(params->train_params.lr_intrinsics));
        }

        if (!g.empty())
        {
            structure_optimizer = std::make_shared<torch::optim::SGD>(g, torch::optim::SGDOptions(10));
        }
    }
    {
        std::vector<torch::optim::OptimizerParamGroup> g_sgd, g_adam;

        if (params->photo_calib_params.exposure_enable)
        {
            std::cout << "Optimizing Tone Mapper Exposure LR " << params->photo_calib_params.exposure_lr << std::endl;
            std::vector<torch::Tensor> t;
            t.push_back(tone_mapper->exposure_bias);
            g_sgd.emplace_back(t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));

            if (params->photo_calib_params.exposure_mult)
            {
                std::cout << "Optimizing Tone Mapper Exposure Factor LR " << params->photo_calib_params.exposure_lr
                          << std::endl;
                t.clear();
                t.push_back(tone_mapper->exposure_factor);
                if (params->photo_calib_params.exposure_sgd)
                {
                    g_sgd.emplace_back(
                        t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));
                }
                else
                {
                    g_adam.emplace_back(
                        t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));
                }
            }
        }
        if (params->photo_calib_params.sensor_bias_enable)
        {
            std::cout << "Optimizing Tone Mapper Sensor Bias LR " << params->photo_calib_params.sensor_bias_lr
                      << std::endl;
            std::vector<torch::Tensor> t;
            t.push_back(tone_mapper->sensor_bias);
            if (params->photo_calib_params.sensor_bias_sgd)
            {
                g_sgd.emplace_back(
                    t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.sensor_bias_lr));
            }
            else
            {
                g_adam.emplace_back(
                    t, std::make_unique<torch::optim::AdamOptions>(params->photo_calib_params.sensor_bias_lr));
            }
        }
        if (tone_mapper->response)
        {
            std::cout << "Optimizing Sensor Response Bias LR " << params->photo_calib_params.response_lr << std::endl;
            std::vector<torch::Tensor> t = tone_mapper->response->parameters();

            if (params->photo_calib_params.response_sgd)
            {
                g_sgd.emplace_back(t,
                                   std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.response_lr));
            }
            else
            {
                g_adam.emplace_back(
                    t, std::make_unique<torch::optim::AdamOptions>(params->photo_calib_params.response_lr));
            }
        }
        if (!g_sgd.empty())
        {
            tm_optimizer_sgd = std::make_shared<torch::optim::SGD>(g_sgd, torch::optim::SGDOptions(10));
        }
        if (!g_adam.empty())
        {
            tm_optimizer_adam = std::make_shared<torch::optim::Adam>(g_adam, torch::optim::AdamOptions(10));
        }
    }

    std::sort(train_indices.begin(), train_indices.end());
    std::sort(test_indices.begin(), test_indices.end());
    active_train_images =
        torch::from_blob(train_indices.data(), {(long)train_indices.size()}, torch::TensorOptions(torch::kInt32))
            .clone()
            .cuda();
    active_test_images =
        torch::from_blob(test_indices.data(), {(long)test_indices.size()}, torch::TensorOptions(torch::kInt32))
            .clone()
            .cuda();
}


void SceneBase::Finalize2()
{
    std::vector<SE3> poses;
    for (auto& f : frames)
    {
        poses.push_back(f->pose);
    }
    std::vector<IntrinsicsPinholed> intrinsics;
    for (auto c : cameras)
    {
        intrinsics.push_back(c.K);
    }
    pose         = CameraPoseModule(poses);
    camera_model = CameraModelModule(cameras.front().h, cameras.front().w, intrinsics);
    vol_translate_para =  vol_translate_para.to(device);

    pose->to(device);
    camera_model->to(device);

    CHECK(params);
    CHECK(camera_model);
    tone_mapper = PhotometricCalibration(frames.size(), cameras.size(), camera_model->h, camera_model->w,
                                         params->photo_calib_params);
    tone_mapper->to(device);

    {
        std::vector<torch::optim::OptimizerParamGroup> g;
        if (params->train_params.optimize_pose)
        {
            g.emplace_back(pose->parameters(),
                           std::make_unique<torch::optim::SGDOptions>(params->train_params.lr_pose));
        }
        if (params->train_params.optimize_intrinsics)
        {
            g.emplace_back(camera_model->parameters(),
                           std::make_unique<torch::optim::SGDOptions>(params->train_params.lr_intrinsics));
        }

        if (!g.empty())
        {
            structure_optimizer = std::make_shared<torch::optim::SGD>(g, torch::optim::SGDOptions(10));
        }
    }
    {
        std::vector<torch::optim::OptimizerParamGroup> g_sgd, g_adam;

        if (params->photo_calib_params.exposure_enable)
        {
            std::cout << "Optimizing Tone Mapper Exposure LR " << params->photo_calib_params.exposure_lr << std::endl;
            std::vector<torch::Tensor> t;
            t.push_back(tone_mapper->exposure_bias);
            g_sgd.emplace_back(t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));

            if (params->photo_calib_params.exposure_mult)
            {
                std::cout << "Optimizing Tone Mapper Exposure Factor LR " << params->photo_calib_params.exposure_lr
                          << std::endl;
                t.clear();
                t.push_back(tone_mapper->exposure_factor);
                if (params->photo_calib_params.exposure_sgd)
                {
                    g_sgd.emplace_back(
                        t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));
                }
                else
                {
                    g_adam.emplace_back(
                        t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.exposure_lr));
                }
            }
        }
        if (params->photo_calib_params.sensor_bias_enable)
        {
            std::cout << "Optimizing Tone Mapper Sensor Bias LR " << params->photo_calib_params.sensor_bias_lr
                      << std::endl;
            std::vector<torch::Tensor> t;
            t.push_back(tone_mapper->sensor_bias);
            if (params->photo_calib_params.sensor_bias_sgd)
            {
                g_sgd.emplace_back(
                    t, std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.sensor_bias_lr));
            }
            else
            {
                g_adam.emplace_back(
                    t, std::make_unique<torch::optim::AdamOptions>(params->photo_calib_params.sensor_bias_lr));
            }
        }
        if (tone_mapper->response)
        {
            std::cout << "Optimizing Sensor Response Bias LR " << params->photo_calib_params.response_lr << std::endl;
            std::vector<torch::Tensor> t = tone_mapper->response->parameters();

            if (params->photo_calib_params.response_sgd)
            {
                g_sgd.emplace_back(t,
                                   std::make_unique<torch::optim::SGDOptions>(params->photo_calib_params.response_lr));
            }
            else
            {
                g_adam.emplace_back(
                    t, std::make_unique<torch::optim::AdamOptions>(params->photo_calib_params.response_lr));
            }
        }
        if (!g_sgd.empty())
        {
            tm_optimizer_sgd = std::make_shared<torch::optim::SGD>(g_sgd, torch::optim::SGDOptions(10));
        }
        if (!g_adam.empty())
        {
            tm_optimizer_adam = std::make_shared<torch::optim::Adam>(g_adam, torch::optim::AdamOptions(10));
        }
    }

    std::sort(train_indices.begin(), train_indices.end());
    std::sort(test_indices.begin(), test_indices.end());
    active_train_images =
        torch::from_blob(train_indices.data(), {(long)train_indices.size()}, torch::TensorOptions(torch::kInt32))
            .clone()
            .cuda();
    active_test_images =
        torch::from_blob(test_indices.data(), {(long)test_indices.size()}, torch::TensorOptions(torch::kInt32))
            .clone()
            .cuda();
}

RayList SceneBase::GetRays(torch::Tensor uv, torch::Tensor image_id, torch::Tensor camera_id)
{
    RayList result;
    result.Allocate(uv.size(0), 3);
    result.to(uv.device());


    auto px_coords = UVToPixel(uv, camera_model->w, camera_model->h, uv_align_corners);


    if (dataset_params.camera_model == "pinhole")
    {
        auto unproj =
            camera_model->Unproject(camera_id, px_coords, torch::ones({uv.size(0)}, torch::TensorOptions(uv.device())));
        unproj = unproj / torch::norm(unproj, 2, 1, true);

        auto dir2 = pose->RotatePoint(unproj, image_id);
        CHECK_EQ(dir2.requires_grad(), unproj.requires_grad());

        result.origin    = torch::index_select(pose->translation, 0, image_id).to(torch::kFloat32);
        result.direction = dir2;
        torch::Tensor unique_ori = std::get<0>(at::_unique(result.origin));
    } 
    else if (dataset_params.camera_model == "orthographic")
    {
        // printf("use para beam\n");
        auto depth  = torch::ones({uv.size(0)}, torch::TensorOptions(uv.device()));
        auto unproj = camera_model->Unproject(camera_id, px_coords, depth);

        unproj = torch::cat({unproj.slice(1, 0, 2), torch::zeros_like(depth).unsqueeze(1)}, 1);
        //        PrintTensorInfo(unproj);


        auto dir2 = pose->RotatePoint(unproj, image_id);

        // [65536 ,3]
        // printf("dir 2 is");
        // PrintTensorInfo(dir2);
        // auto dir3 = pose->TranslatePoint(dir2, image_id);

        auto dir3 = dir2 + vol_translate_para.unsqueeze(0);



        // std::cout << dir2[0][0].item<double>() << " " <<dir2[0][1].item<double>() << " " <<dir2[0][2].item<double>()<< std::endl;


        //     PrintTensorInfo(unproj.slice(1,0,1));
        //     PrintTensorInfo(unproj.slice(1,1,2));
        //     PrintTensorInfo(unproj.slice(1,2,3));
        
        //         PrintTensorInfo(dir2);

        auto direction = torch::cat({torch::zeros_like(depth).unsqueeze(1), torch::zeros_like(depth).unsqueeze(1),
                                     torch::ones_like(depth).unsqueeze(1)},
                                    1)
                             .to(torch::kDouble);
        direction = pose->RotatePoint(direction, image_id);


        // auto shift = torch::cat({torch::zeros_like(depth).unsqueeze(1), 0.2 * torch::ones_like(depth).unsqueeze(1),
        //                          torch::zeros_like(depth).unsqueeze(1)},
        //                     1)
        //                 .to(torch::kDouble);

        result.direction = direction;
        // result.origin    = dir2 - (direction * (sqrt(2.0) )) + shift;

        result.origin    = dir3 - (direction * (sqrt(2.0) ));

    }
    else
    {
        CHECK(false);
    }

    result.origin    = result.origin.to(torch::kFloat32);
    result.direction = result.direction.to(torch::kFloat32);

    return result;
}


RayList SceneBase::GetRays(torch::Tensor uv, torch::Tensor image_id, torch::Tensor camera_id, torch::Tensor defocus)
{
    RayList result;
    result.Allocate(uv.size(0), 3);
    result.to(uv.device());


    auto px_coords = UVToPixel(uv, camera_model->w, camera_model->h, uv_align_corners);


    if (dataset_params.camera_model == "pinhole")
    {
        auto unproj =
            camera_model->Unproject(camera_id, px_coords, torch::ones({uv.size(0)}, torch::TensorOptions(uv.device())));
        unproj = unproj / torch::norm(unproj, 2, 1, true);

        auto dir2 = pose->RotatePoint(unproj, image_id);
        CHECK_EQ(dir2.requires_grad(), unproj.requires_grad());

        result.origin    = torch::index_select(pose->translation, 0, image_id).to(torch::kFloat32);
        result.direction = dir2;
    }
    else if (dataset_params.camera_model == "orthographic")
    {
        // printf("use para beam\n");
        auto depth  = torch::ones({uv.size(0)}, torch::TensorOptions(uv.device()));
        auto unproj = camera_model->Unproject(camera_id, px_coords, depth);

        unproj = torch::cat({unproj.slice(1, 0, 2), torch::zeros_like(depth).unsqueeze(1)}, 1);
        //        PrintTensorInfo(unproj);


        auto dir2 = pose->RotatePoint(unproj, image_id);

        // auto dir3 = pose->TranslatePoint(dir2, image_id);

        auto dir3 = dir2 + vol_translate_para.unsqueeze(0);



        // std::cout << dir2[0][0].item<double>() << " " <<dir2[0][1].item<double>() << " " <<dir2[0][2].item<double>()<< std::endl;


        //     PrintTensorInfo(unproj.slice(1,0,1));
        //     PrintTensorInfo(unproj.slice(1,1,2));
        //     PrintTensorInfo(unproj.slice(1,2,3));
        
        //         PrintTensorInfo(dir2);

        auto direction = torch::cat({torch::zeros_like(depth).unsqueeze(1), torch::zeros_like(depth).unsqueeze(1),
                                     torch::ones_like(depth).unsqueeze(1)},
                                    1)
                             .to(torch::kDouble);
        direction = pose->RotatePoint(direction, image_id);


        // auto shift = torch::cat({torch::zeros_like(depth).unsqueeze(1), 0.2 * torch::ones_like(depth).unsqueeze(1),
        //                          torch::zeros_like(depth).unsqueeze(1)},
        //                     1)
        //                 .to(torch::kDouble);

        result.direction = direction;

        torch::Tensor scale_tensor, dir_orth;
        if(params->train_params.use_defocus && defocus.defined())
        {
            #if 0
            auto dir_orth = torch::zeros_like(direction, torch::TensorOptions(uv.device())).uniform_(0,100);
            auto index_replace = torch::randint( 0, 3, {dir_orth.sizes()[0]}, torch::TensorOptions(uv.device())).unsqueeze(1).to(torch::kInt64);
            auto index_re = torch::linspace(0,dir_orth.sizes()[0]-1, {dir_orth.sizes()[0]}, torch::TensorOptions(uv.device())).unsqueeze(1).to(torch::kInt64);
            index_re = torch::cat({index_re, index_replace}, 1).t();

            // auto t1 = torch::arange(3, torch::TensorOptions(uv.device())).expand({dir_orth.sizes()[0],-1});

            // PrintTensorInfo(direction.gather(1, t1==index_replace));
            printf("test index ");
            PrintTensorInfo(direction.gather(1, index_replace));
            // index select TO DEBUG
            auto replace_val = direction.gather(1, index_replace);
            PrintTensorInfo(replace_val);
            auto missing_val = - replace_val * replace_val/(replace_val + 1e-10);
            printf("missing val ");
            PrintTensorInfo(missing_val);
            PrintTensorInfo(direction);
            // direction.gather(1, index_replace) = missing_val;
            // direction.index_put_({index_re}, missing_val);
            // change here

            PrintTensorInfo(direction);
            PrintTensorInfo(direction.gather(1, index_replace));
            std::cout << std::endl;
            // printf("norma");
            // PrintTensorInfo(torch::nn::functional::normalize(direction,torch::nn::functional::NormalizeFuncOptions().p(2).dim(1)));
            // PrintTensorInfo(torch::norm(direction,2,1));
            // PrintTensorInfo(torch::normalize(direction, 1));

            direction = direction/torch::norm(direction,2,1).unsqueeze(1);

            auto test = torch::nn::functional::normalize(direction,torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
            printf("direction is ");
            PrintTensorInfo(direction);
            PrintTensorInfo(torch::norm(direction,2,1).unsqueeze(1));
            PrintTensorInfo(test);
            auto kernel_size = defocus * 2/ camera_model->w;
            PrintTensorInfo(kernel_size);
            std::cout << " value " << params->train_params.defocus_std_init <<" width " << camera_model->w << std::endl;
            float std_value = 2 * params->train_params.defocus_std_init/ camera_model->w;
            printf("test tt\n");
            // auto size_value = torch::normal(0, std_value, kernel_size.sizes(), torch::TensorOptions(uv.device()));
            auto size_value = torch::zeros({kernel_size.sizes()}, torch::TensorOptions(uv.device()));
            size_value.normal_(0, std_value);

            // size_value.normal_(0, std_value);
            printf("kernel value ");
            PrintTensorInfo(size_value);
            PrintTensorInfo(kernel_size);
            scale = torch::cat({kernel_size.unsqueeze(1), size_value.unsqueeze(1)},1);
            PrintTensorInfo(scale);
            printf("min value ");
            // PrintTensorInfo(scale.max(1));
            // auto [scale_min, min_index] = scale.min(1);
            // (void) min_index;
            // // PrintTensorInfo(per_node_max_density);
            // scale = scale_min;
            // new 
            #else
                dir_orth = torch::zeros_like(direction, torch::TensorOptions(uv.device())).uniform_(0,100);
                dir_orth =  torch::nn::functional::normalize(dir_orth,torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
                // printf("test here \n");
                // PrintTensorInfo(dir_orth);
                auto kernel_size = defocus * 2/ camera_model->w;
                float std_value = 2 * params->train_params.defocus_std_init/ camera_model->w;
                auto size_value = torch::zeros({kernel_size.sizes()}, torch::TensorOptions(uv.device()));
                size_value.normal_(0, std_value);
                auto scale = torch::cat({kernel_size.unsqueeze(1), size_value.unsqueeze(1)},1);
                auto [scale_min, min_index] = scale.min(1);
                (void) min_index;
                scale_tensor = scale_min;
            // PrintTensorInfo(scale_min);
            // PrintTensorInfo(dir_orth * scale_tensor.unsqueeze(1));
            #endif
        }

        // result.origin    = dir2 - (direction * (sqrt(2.0) )) + shift;
        if(params->train_params.use_defocus && defocus.defined())
        {
            result.origin    = dir3 - (direction * (sqrt(2.0) )) + dir_orth * scale_tensor.unsqueeze(1) ;
        }
        else
        {
            result.origin    = dir3 - (direction * (sqrt(2.0) ));
        }
    }
    else
    {
        CHECK(false);
    }

    result.origin    = result.origin.to(torch::kFloat32);
    result.direction = result.direction.to(torch::kFloat32);

    return result;
}

void SceneBase::PrintGradInfo(int epoch_id, TensorBoardLogger* logger)
{
    {
        auto t = pose->rotation_tangent;
        std::vector<double> mean;

        if (t.grad().defined())
        {
            mean.push_back(t.grad().abs().mean().item().toFloat());
        }

        logger->add_scalar("Gradient/" + scene_name + "/rotation", epoch_id, Statistics(mean).mean);
    }
    {
        auto t = pose->translation;
        std::vector<double> mean;

        if (t.grad().defined())
        {
            mean.push_back(t.grad().abs().mean().item().toFloat());
        }

        logger->add_scalar("Gradient/" + scene_name + "/translation", epoch_id, Statistics(mean).mean);
    }
    {
        auto params = camera_model->parameters();
        std::vector<double> mean;
        for (auto t : params)
        {
            if (t.grad().defined())
            {
                mean.push_back(t.grad().abs().mean().item().toFloat());
            }
        }
        logger->add_scalar("Gradient/" + scene_name + "/intrinsics", epoch_id, Statistics(mean).mean);
    }
}
void SceneBase::OptimizerStep(int epoch_id, bool only_image_params)
{
    if (structure_optimizer)
    {
        if (epoch_id > params->train_params.optimize_structure_after_epochs)
        {
            structure_optimizer->step();
            pose->ApplyTangent();
            // pose->ApplyTranslate();
        }
        structure_optimizer->zero_grad();
    }

    if (only_image_params)
    {
        if (tone_mapper->params.sensor_bias_enable)
        {
            tone_mapper->sensor_bias.mutable_grad().zero_();
        }
        if (tone_mapper->response)
        {
            tone_mapper->response->response.mutable_grad().zero_();
        }
    }
    if (tm_optimizer_sgd)
    {
        if (epoch_id > params->train_params.optimize_tone_mapper_after_epochs)
        {
            tm_optimizer_sgd->step();
        }
        tm_optimizer_sgd->zero_grad();
    }
    if (tm_optimizer_adam)
    {
        if (epoch_id > params->train_params.optimize_tone_mapper_after_epochs)
        {
            tm_optimizer_adam->step();
        }
        tm_optimizer_adam->zero_grad();
    }

    tone_mapper->ApplyConstraints();
}
void SceneBase::PrintInfo(int epoch_id, TensorBoardLogger* logger)
{
    std::cout << std::left;
    if (params->train_params.optimize_pose)
    {
        // std::cout << std::setw(30) << "Pose T: " << TensorInfo(pose->translation) << std::endl;
        // std::cout << std::setw(30) << "Pose R:    " << TensorInfo(pose->rotation) << std::endl;
    }
    if (params->train_params.optimize_intrinsics)
    {
        // std::cout << std::setw(30) << "Intrinsics: " << camera_model->DownloadPinhole().front() << std::endl;
    }

    if (tone_mapper->params.exposure_enable && tone_mapper->params.exposure_lr > 0)
    {
        auto selected_bias_train = torch::index_select(tone_mapper->exposure_bias, 0, active_train_images).cpu();
        auto selected_bias_test  = torch::index_select(tone_mapper->exposure_bias, 0, active_test_images).cpu();
        for (int i = 0; i < selected_bias_train.size(0); ++i)
        {
            logger->add_scalar("ToneMapper/" + scene_name + "/exp_bias_train", i,
                               selected_bias_train.data_ptr<float>()[i]);
        }
        for (int i = 0; i < selected_bias_test.size(0); ++i)
        {
            logger->add_scalar("ToneMapper/" + scene_name + "/exp_bias_test", i,
                               selected_bias_test.data_ptr<float>()[i]);
        }
    }
    if (tone_mapper->params.sensor_bias_enable && tone_mapper->params.sensor_bias_lr > 0)
    {
        auto bias_image = tone_mapper->sensor_bias;
        // std::cout << std::setw(30) << "Sensor Bias:" << TensorInfo(bias_image) << std::endl;

        auto err_col_tens = ColorizeTensor(bias_image.squeeze(0).squeeze(0) * 8, colorizeTurbo);
        LogImage(logger, TensorToImage<ucvec3>(err_col_tens), "Tonemapper/" + scene_name + "/sensor_bias_x8", epoch_id);
    }


    if (tone_mapper->response)
    {
        auto crfs = tone_mapper->response->GetCRF().front();
        LogImage(logger, crfs.Image(), "Tonemapper/" + scene_name + "/sensor_response", epoch_id);
    }
}
SceneBase::SceneBase(std::string _scene_dir)
{
    scene_path = std::filesystem::canonical(_scene_dir).string();
    scene_name = std::filesystem::path(scene_path).filename();

    std::cout << "====================================" << std::endl;
    std::cout << "Scene Base" << std::endl;
    std::cout << "  Name         " << scene_name << std::endl;
    std::cout << "  Path         " << scene_path << std::endl;
    SAIGA_ASSERT(!scene_name.empty());
    CHECK(std::filesystem::exists(scene_path));
    CHECK(std::filesystem::exists(scene_path + "/dataset.ini"));

    auto file_pose           = scene_path + "/poses.txt";
    auto file_image_names    = scene_path + "/images.txt";
    auto file_mask_names     = scene_path + "/masks.txt";
    auto file_camera_indices = scene_path + "/camera_indices.txt";

    dataset_params = DatasetParams(scene_path + "/dataset.ini");

    {
        CameraBase cam(scene_path + "/camera.ini");
        CHECK_GT(cam.w, 0);
        CHECK_GT(cam.h, 0);
        cameras.push_back(cam);
    }


    if (!dataset_params.volume_file.empty())
    {
        CHECK(std::filesystem::exists(scene_path + "/" + dataset_params.volume_file));
        torch::load(ground_truth_volume, scene_path + "/" + dataset_params.volume_file);
        std::cout << "Ground Truth Volume " << TensorInfo(ground_truth_volume) << std::endl;


        // save as hdr
        // SaveHDRImageTensor(ground_truth_volume, scene_name + "_reference.hdr");
    }


    std::vector<Sophus::SE3d> poses;
    if (std::filesystem::exists(file_pose))
    {
        std::ifstream strm(file_pose);

        std::string line;
        while (std::getline(strm, line))
        {
            std::stringstream sstream(line);
            Quat q;
            Vec3 t;
            sstream >> q.x() >> q.y() >> q.z() >> q.w() >> t.x() >> t.y() >> t.z();
            poses.push_back({q, t});
        }
    }


    std::vector<std::string> images;
    std::vector<std::string> masks;

    if (std::filesystem::exists(file_image_names))
    {
        std::ifstream strm(file_image_names);

        std::string line;
        while (std::getline(strm, line))
        {
            images.push_back(line);
        }
    }

    if (std::filesystem::exists(file_mask_names))
    {
        std::ifstream strm(file_mask_names);

        std::string line;
        while (std::getline(strm, line))
        {
            masks.push_back(line);
        }
    }

    std::vector<int> camera_indices;
    if (std::filesystem::exists(file_camera_indices))
    {
        std::ifstream strm(file_camera_indices);

        std::string line;
        while (std::getline(strm, line))
        {
            camera_indices.push_back(to_int(line));
        }
    }

    int n_frames = std::max({images.size(), poses.size()});
    frames.resize(n_frames);

    SAIGA_ASSERT(!poses.empty());
    SAIGA_ASSERT(masks.empty() || masks.size() == frames.size());
    SAIGA_ASSERT(camera_indices.empty() || camera_indices.size() == frames.size());

    for (int i = 0; i < n_frames; ++i)
    {
        int camera_id = camera_indices.empty() ? 0 : camera_indices[i];
        auto img      = std::make_shared<UnifiedImage>();

        img->camera_id          = camera_id;
        img->pose               = poses[i];
        img->pose.translation() = img->pose.translation() * dataset_params.scene_scale;
        if (!images.empty()) img->image_file = images[i];

        if (!masks.empty()) img->mask_file = masks[i];

        frames[i] = img;
    }

    std::cout << "test scene 0 " << std::endl;
    std::vector<double> distances;
    for (auto& img : frames)
    {
        auto d = img->pose.translation().norm();
        distances.push_back(d);
    }
    std::cout << "  Avg distance " << Statistics(distances).mean << std::endl;

    std::cout << "  Volume Scale " << dataset_params.scene_scale << std::endl;
    std::cout << "  Images       " << frames.size() << std::endl;
    std::cout << "  Img. Size    " << cameras.front().w << " x " << cameras.front().h << std::endl;
    std::cout << "====================================" << std::endl;
}

torch::Tensor SceneBase::LoadImageRaw(std::string file)
{
    CHECK(std::filesystem::exists(file));


    Image raw(file);
    TemplatedImage<float> raw_float(raw.dimensions());
    if (raw.type == Saiga::US1)
    {
        auto raw_view = raw.getImageView<unsigned short>();
        for (auto i : raw_view.rowRange())
        {
            for (auto j : raw_view.colRange())
            {
                raw_float(i, j) = raw_view(i, j);
            }
        }
    }
    else if (raw.type == Saiga::F1)
    {
        auto raw_view = raw.getImageView<float>();
        raw_view.copyTo(raw_float.getImageView());
    }
    else
    {
        CHECK(false) << "image type " << raw.type << " not implemented" << std::endl;
    }

    CHECK_EQ(raw_float.h, cameras[0].h);
    CHECK_EQ(raw_float.w, cameras[0].w);
    return ImageViewToTensor(raw_float.getImageView());
}

void SceneBase::LoadImagesCT(std::vector<int> indices)
{
    std::cout << "log input " << dataset_params.log_space_input
              << " Log10 conversion: " << dataset_params.use_log10_conversion << std::endl;

        //new add laplacian process

    for(int i = 0; i < dataset_params.projection_scale; ++i)
    {
        std::string output_tmpname = dataset_params.image_dir + "/image_scale_"+to_string(i);
        std::filesystem::create_directories(output_tmpname);
    }

    ProgressBar bar(std::cout, "Load images", indices.size());
    for (int i = 0; i < indices.size(); ++i)
    {
        int image_id = indices[i];
        auto& img    = frames[image_id];
        if (img->projection.defined()) continue;

        std::string file_extension = (img->image_file.substr(img->image_file.find_last_of(".") + 1));
        // std::cout << "file extension " << file_extension << std::endl;
        torch::Tensor raw_tensor;
        if( file_extension == "npz")
        {
            raw_tensor = loadNumpyNpzIntoTensor(dataset_params.image_dir + "/" + img->image_file, "proj");
        }
        else if (file_extension == "npy")
        {
            raw_tensor = loadNumpyIntoTensor(dataset_params.image_dir + "/" + img->image_file);

        }
        else
        {
            raw_tensor = LoadImageRaw(dataset_params.image_dir + "/" + img->image_file);
        }

        if (dataset_params.log_space_input)
        {
            img->projection = raw_tensor / dataset_params.xray_max;
        }
        else
        {
            // log of negative numbers/zero is undefined
            raw_tensor = raw_tensor.clamp_min(1e-5);
            // printf("conduct log conversion\n");

            // log10 / loge conversion
            if (dataset_params.use_log10_conversion)
            {
                // printf("use log 10 conversion\n");
                // convert transmittance to absorption
                img->projection = -torch::log10(raw_tensor / dataset_params.xray_max);

                img->projection = img->projection * (dataset_params.projection_factor /
                                                     -std::log10(dataset_params.xray_min / dataset_params.xray_max));
            }
            else
            {
                // convert transmittance to absorption
                img->projection = -torch::log(raw_tensor / dataset_params.xray_max);

                img->projection = img->projection * (dataset_params.projection_factor /
                                                     -std::log(dataset_params.xray_min / dataset_params.xray_max));
            }
        }

        CHECK(img->projection.isfinite().all().item().toBool())
            << "After log conversion projection " << img->image_file << " is not finite. Probably because of log(0)";


        if (!img->mask_file.empty() && !dataset_params.mask_dir.empty())
        {
            auto mask_file = dataset_params.mask_dir + "/" + img->mask_file;
            CHECK(std::filesystem::exists(mask_file)) << mask_file;

            TemplatedImage<unsigned char> mask(mask_file);
            img->mask = ImageViewToTensor(mask.getImageView());

            img->projection = img->projection * img->mask;
        }
        img->projections.push_back(img->projection);
        if (!img->mask_file.empty() && !dataset_params.mask_dir.empty())
        {
            img->masks.push_back(img->mask);
        }
    // if (!img->mask_file.empty() && dataset_params.mask_dir.empty())
    // {
    //     printf("---------MASK ENABLED but could not find mask------------------\n");
    // }
        // std::cout<<"proj size " << std::endl;
        // PrintTensorInfo(img->projection);


        auto proj = img->projections[0];

        std::string out_imagesavename = (img->image_file.substr(img->image_file.find_last_of(".")));
        std::string output_tmpname = dataset_params.image_dir + "/image_scale_"+to_string(0) + "/" + out_imagesavename + ".tiff";
        auto im1 = TensorToImage<float>(proj);
        im1.save(output_tmpname);

        for(int j = 1; j < dataset_params.projection_scale; ++j)
        {
            auto proj = laplacian2d->downsample(img->projections[j-1].unsqueeze(0));

            img->projections.push_back(proj.squeeze(0));

            std::string output_tmpname = dataset_params.image_dir + "/image_scale_"+to_string(j) + "/" + out_imagesavename+ ".tiff";
            auto im1 = TensorToImage<float>(proj);
            im1.save(output_tmpname);

            if (!img->mask_file.empty() && !dataset_params.mask_dir.empty())
            {
                proj = laplacian2d->downsample(img->masks[j-1].unsqueeze(0));
                img->masks.push_back(proj.squeeze(0));
            }
        }

        bar.addProgress(1);
    }
    // printf("current camera is %d", cameras.size());
}

void SceneBase::setcurrentproj(int scale,std::vector<int> indices)
{
    // ProgressBar bar(std::cout, "Setting images", indices.size());
    // for (int i = 0; i < indices.size(); ++i)
    // {
    //     int image_id = indices[i];
    //     auto& img    = frames[image_id];
    //     if (img->projection.defined()) continue;

    //     img->projection = img->projections[scale];
    //     img->mask       = img->masks[scale];

    //     bar.addProgress(1);

    // }

    float downfactor = std::pow(2,scale);
    // printf("current camera is %d", cameras.size());
    // std::cout << "current camera is " << cameras.size() << std::endl;
    // for (int i = 0; i < cameras.size();++i)
    // {
    //     CameraBase cam1;
    //     cam1.w = ori_w/downfactor;
    //     cam1.h = ori_h/downfactor;
    //     cam1.K.cx = cam1.w/2.;
    //     cam1.K.cy = cam1.h/2.;
    //     cam1.K.fx = cam1.K.cx * dataset_params.camera_proj_scale;
    //     cam1.K.fy = cam1.K.cy * dataset_params.camera_proj_scale;
    //     std::cout <<" camera is " << cam1.w <<" " << cam1.h << " " << cam1.K.cx <<" "<<cam1.K.fx<< std::endl;
    //     cameras[i] = cam1;
    // }
    // Finalize2();

    params->train_params.rays_per_image = rays_per_image_level0/downfactor/downfactor;

    std::cout << "rays_per_image " << rays_per_image_level0 << " " << params->train_params.rays_per_image << " " <<(2^scale)<<std::endl;

}


void SceneBase::setmoment(std::vector<torch::Tensor> projection_images,std::vector<int> indices, float factor, std::string save_file_name)
{
    int out_w = projection_images[0].size(1);
    int out_h = projection_images[0].size(2);
    printf("out put proj size %d %d\n", out_w, out_h);
    torch::Tensor reprojection   = torch::empty({0, out_h, out_w });
    torch::Tensor ori_projection = torch::empty({0, out_h, out_w });
    torch::Tensor residual       = torch::empty({0, out_h, out_w });



    torch::Tensor image_input_new = torch::empty({0, out_h, out_w});
    for (int i = 0; i < indices.size(); ++i)
    {
        int image_id = indices[i];
        auto& img    = frames[image_id];
        if (!img->projection.defined()) continue;
        reprojection    = torch::cat({reprojection, projection_images[i]},0);
        ori_projection  = torch::cat({ori_projection, img->projection}, 0);
        residual        = torch::cat({residual, torch::abs(projection_images[i] - img->projection)}, 0);

        img->projection = factor * img->projection + (1- factor) * projection_images[i];

        image_input_new = torch::cat({image_input_new, img->projection}, 0);
        // img->mask       = img->masks[scale];
    }


    torch::Tensor image_to_store = torch::cat({ori_projection, reprojection, residual}, 2);
    // for(int i : indices)
    // {
    //     reprojection = torch::cat({reprojection, projection_images[i]}, 0);
    // }
    std::string save_file_name1 = save_file_name + ".hdr";
    SaveHDRImageTensor(image_to_store.unsqueeze(0), save_file_name1);

    std::string save_file_name2 = save_file_name + "_input.hdr";
    SaveHDRImageTensor(image_input_new.unsqueeze(0), save_file_name2);
}


void SceneBase::Draw(TensorBoardLogger* logger)
{
    std::string log_name = "Input/" + scene_name + "/";

    double max_radius = 0;
    for (auto& img_ : frames)
    {
        auto img   = (img_);
        max_radius = std::max(img->pose.translation().norm(), max_radius);
    }

    double scale = 1.0 / max_radius * 0.9;


    std::cout << "Drawing cone scene 2d scale = " << scale << std::endl;
    TemplatedImage<ucvec3> target_img(256, 256);
    target_img.makeZero();

    auto normalized_to_target = [&](Vec2 p) -> Vec2
    {
        p = p * scale;
        p = ((p + Vec2::Ones()) * 0.5).array() * Vec2(target_img.h - 1, target_img.w - 1).array();

        Vec2 px = p.array().round().cast<double>();

        std::swap(px(0), px(1));
        return px;
    };


    auto draw_normalized_line = [&](auto& img, Vec2 p1, Vec2 p2, ucvec3 color)
    {
        Vec2 p11 = normalized_to_target(p1);
        Vec2 p22 = normalized_to_target(p2);

        std::swap(p11(0), p11(1));
        std::swap(p22(0), p22(1));

        ImageDraw::drawLineBresenham(img.getImageView(), p11.cast<float>(), p22.cast<float>(), color);
    };

    auto draw_normalized_circle = [&](auto& img, Vec2 p1, double r, ucvec3 color)
    {
        auto p = normalized_to_target(p1);
        ImageDraw::drawCircle(img.getImageView(), vec2(p(1), p(0)), r, color);
    };



    auto draw_axis = [&](auto& img)
    {
        int rad = 2;
        for (int r = -rad; r <= rad; ++r)
        {
            Vec2 p1 = normalized_to_target(Vec2(0, -1)) + Vec2(r, 0);
            Vec2 p2 = normalized_to_target(Vec2(0, 1)) + Vec2(r, 0);

            Vec2 p3 = normalized_to_target(Vec2(-1, 0)) + Vec2(0, r);
            Vec2 p4 = normalized_to_target(Vec2(1, 0)) + Vec2(0, r);

            ImageDraw::drawLineBresenham(img.getImageView(), p1.cast<float>(), p2.cast<float>(), ucvec3(0, 255, 0));
            ImageDraw::drawLineBresenham(img.getImageView(), p3.cast<float>(), p4.cast<float>(), ucvec3(255, 0, 0));
        }

        draw_normalized_line(img, Vec2(-1, -1), Vec2(1, -1), ucvec3(255, 255, 255));
        draw_normalized_line(img, Vec2(-1, -1), Vec2(-1, 1), ucvec3(255, 255, 255));
        draw_normalized_line(img, Vec2(1, 1), Vec2(1, -1), ucvec3(255, 255, 255));
        draw_normalized_line(img, Vec2(1, 1), Vec2(-1, 1), ucvec3(255, 255, 255));
    };


    int log_count = 0;
    {
        // draw some debug stuff
        auto img_cpy = target_img;
        draw_axis(img_cpy);

        for (int i = 0; i < frames.size(); ++i)
        {
            auto img     = frames[i];
            ucvec3 color = ucvec3(255, 255, 255);

            if (std::find(train_indices.begin(), train_indices.end(), i) != train_indices.end())
            {
                color = ucvec3(0, 255, 0);
            }
            else if (std::find(test_indices.begin(), test_indices.end(), i) != test_indices.end())
            {
                color = ucvec3(255, 0, 0);
            }
            else
            {
                continue;
            }


            draw_normalized_circle(img_cpy, img->pose.translation().head<2>(), 5, color);


            if (i % 1 == 0)
            {
                auto img_cpy = target_img;
                draw_axis(img_cpy);
                //                draw_normalized_circle(img_cpy, img->pose.translation().head<2>(), 5, color);


                auto uv = img->CoordinatesRandom(50).to(device);
                auto image_id =
                    torch::full({uv.size(0)}, (int)i, torch::TensorOptions(torch::kInt32).device(uv.device()));
                auto camera_id = torch::full({uv.size(0)}, (int)img->camera_id,
                                             torch::TensorOptions(torch::kInt32).device(uv.device()));
                auto rays      = GetRays(uv, image_id, camera_id);
                rays.to(torch::kCPU);

                vec3* origin    = rays.origin.data_ptr<vec3>();
                vec3* direction = rays.direction.data_ptr<vec3>();

                //                std::cout << "img center " << img->pose.translation().transpose() << std::endl;
                for (int j = 0; j < rays.size(); ++j)
                {
                    Vec3 o = origin[j].cast<double>();
                    Vec3 d = direction[j].cast<double>();

                    //                    std::cout << o.transpose() << std::endl;
                    draw_normalized_line(img_cpy, o.head<2>(), (o + d).head<2>(), ucvec3(255, 0, 0));
                    draw_normalized_circle(img_cpy, o.head<2>(), 5, ucvec3(0, 255, 0));
                }

                LogImage(logger, img_cpy, log_name + "rays", i);
                //                PrintTensorInfo(rays.origin);
                //                exit(0);
            }
        }
        // img_cpy.save(output_dir + "geom_all_image_planes.png");
        LogImage(logger, img_cpy, log_name + "overview", log_count++);
    }

    int drawn_images = 0;
    for (int i = 0; i < frames.size() && drawn_images < 5; ++i)
    {
        auto img = frames[i];
        if (!img->projection.defined()) continue;
        if (logger)
        {
            auto proj_tens = img->projection;

            auto proj = TensorToImage<unsigned char>(proj_tens);
            LogImage(logger, proj, log_name + "processed", i);



            auto err_col_tens = ColorizeTensor((img->projection - img->projection.min()).squeeze(0) * 8, colorizeTurbo);
            auto proj_ampli   = TensorToImage<ucvec3>(err_col_tens);
            LogImage(logger, proj_ampli, log_name + "amp_x8_minus_min", i);

            if (img->mask.defined())
            {
                auto mask = TensorToImage<unsigned char>(img->mask);
                LogImage(logger, mask, log_name + "mask", i);
            }

            drawn_images++;
        }
    }
}
void SceneBase::InitializeBiasWithBackground(TensorBoardLogger* logger)
{
    torch::NoGradGuard ngg;
    CHECK(tone_mapper);
    auto new_exp_bias = torch::ones_like(tone_mapper->exposure_bias).cpu() * -1;

    double avg    = 0;
    int avg_count = 0;
    for (int i = 0; i < frames.size(); ++i)
    {
        auto img = frames[i];
        if (!img->projection.defined()) continue;

        // We just take the median of the top left corner.
        auto crop                         = img->projection.slice(1, 0, 64).slice(2, 0, 64);
        float median                      = crop.median().item().toFloat();
        new_exp_bias.data_ptr<float>()[i] = median;
        logger->add_scalar("ToneMapper/" + scene_name + "/exp_bias_init", i, median);
        avg += median;
        avg_count++;
    }



    // set all other frames to the avg bias
    for (int i = 0; i < frames.size(); ++i)
    {
        if (new_exp_bias.data_ptr<float>()[i] < 0)
        {
            new_exp_bias.data_ptr<float>()[i] = avg / avg_count;
        }
    }
    tone_mapper->exposure_bias.set_data(new_exp_bias.to(tone_mapper->exposure_bias.device()));
}
torch::Tensor SceneBase::SampleGroundTruth(torch::Tensor global_coordinates)
{
    CHECK_EQ(global_coordinates.dim(), 2);
    CHECK(ground_truth_volume.defined());
    torch::nn::functional::GridSampleFuncOptions opt;
    opt.align_corners(true).padding_mode(torch::kBorder).mode(torch::kBilinear);

    global_coordinates = global_coordinates.unsqueeze(0).unsqueeze(0).unsqueeze(0);

    //    PrintTensorInfo(global_coordinates);
    //    PrintTensorInfo(ground_truth_volume);

    // [batches, num_features, 1, 1, batch_size]
    auto samples = torch::nn::functional::grid_sample(ground_truth_volume.unsqueeze(0), global_coordinates, opt);

    samples = samples.squeeze(0).squeeze(1).squeeze(1);

    // samples = samples.permute({1, 0});


    return samples;
}
void SceneBase::SaveCheckpoint(const std::string& dir)
{
    auto prefix = dir + "/" + scene_name + "_";
    if (pose)
    {
        torch::save(pose, prefix + "pose.pth");
    }
    if (camera_model)
    {
        torch::save(camera_model, prefix + "camera_model.pth");
    }

    if (tone_mapper)
    {
        torch::save(tone_mapper, prefix + "tone_mapper.pth");
    }
}

torch::Tensor SceneBase::get_batch_data_rays_jitter(torch::Tensor origin, torch::Tensor dir, std::vector<int> indices, int image_w, int image_h)
{

    auto angle_dir = scene_path + "/angles.txt";
    std::vector<float> input_angles;
    if(std::filesystem::exists(angle_dir))
    {
        std::ifstream strm(angle_dir);
        std::string line;
        while(std::getline(strm, line))
        {
            std::stringstream sstream(line);
            float t;
            sstream >> t;
            input_angles.push_back(t);
        }
    }
    
    std::vector<float> angle_indices;
    float max_angle = 0;
    for(int i = 0; i < indices.size(); ++i)
    {
        if(max_angle < input_angles[indices[i]])
        max_angle = input_angles[indices[i]];
        angle_indices.push_back(input_angles[indices[i]]);
    }
    
    float max_blur = 50.0;
    float std_init = 1;

    auto center_dist = torch::absolute(torch::linspace(-1,1,image_w));
    float k = tan(max_angle/max_blur);
    torch::Tensor defocus = torch::empty({0}, torch::TensorOptions(torch::kInt32));
    {
        for(int i = 0; i < indices.size(); ++i)
        {
            defocus = torch::cat({defocus, (tan(angle_indices[i]) * center_dist / k).to(torch::kInt32)},0);
        }
    }
    auto defocus_exp = defocus.expand({-1, image_h}).reshape({-1});

    auto dir_orth = torch::zeros_like(origin).uniform_(0,100);
    // auto index_replace = torch::zeros(origin.sizes()[0], torch::TensorOptions(torch::kInt32));
    // index_replace.randint_(0,3);

    auto index_replace = torch::randint({origin.sizes()[0]},0,3);

    auto missing_val = -dir.index_select(0, index_replace)*dir.index_select(0, index_replace)/(dir.index_select(0, index_replace)+1e-10);
    dir.index_select(0,index_replace) = missing_val;
    dir = dir/torch::norm(dir,1);



    // get_length_by_gaussian
    auto kernel_size = defocus_exp * 2/ image_w;
    auto std_value = 2 * std_init /image_w;
    auto size_value = torch::zeros_like(kernel_size).normal_(0, std_value);
    // auto scale = torch::min(torch::cat({kernel_size, size_value},1),1)[0];
    auto scale = torch::cat({kernel_size, size_value},1).min();
    
    auto pos_patches = origin + dir * scale;

    return pos_patches;
}