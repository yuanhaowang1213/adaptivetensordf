/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "KaustScene.h"

#include "saiga/core/geometry/aabb.h"
#include "saiga/core/util/directory.h"
#include "saiga/vision/torch/ColorizeTensor.h"

#include "utils/utils.h"

#include "tensorboard_logger.h"


KaustSceneLoader::KaustSceneLoader(std::shared_ptr<CombinedParams> params)
{
    CHECK(false);
#if 0
    scene = std::make_shared<SceneBase>(params, 1);
     std::cout << "Loading Kaust Scene " << params->train_params.scene_dir << std::endl;
    // CHECK(false);
    std::string full_dir;  //= params->dataset_params.scene_dir + "/" + params->dataset_params.scene_name;

    full_dir = params->train_params.scene_dir;
    Directory dir(full_dir);

    auto info_files = dir.getFilesEnding(".ctprofile.xml");
    CHECK_EQ(info_files.size(), 1);
    dataset_name = info_files.front().substr(0, info_files.front().size() - 14);
    std::cout << "Dataset name: " << dataset_name << std::endl;

    auto image_names = dir.getFilesEnding(".tif");
    std::sort(image_names.begin(), image_names.end());


    ct_params = XTekCT(full_dir + "/" + dataset_name + ".xtekct");



    image_scale = 1;

    ct_params.DetectorPixelsY *= image_scale;
    ct_params.DetectorPixelsX *= image_scale;
    ct_params.DetectorPixelSizeX /= image_scale;
    ct_params.DetectorPixelSizeY /= image_scale;

    if (params->dataset_params.crop_corner_low(0) < 0)
    {
        params->dataset_params.crop_corner_low  = ivec2(0, 0);
        params->dataset_params.crop_corner_high = ivec2(ct_params.DetectorPixelsX, ct_params.DetectorPixelsY);
    }
    else
    {
        params->dataset_params.crop_corner_low =
            (params->dataset_params.crop_corner_low.cast<float>() * image_scale).cast<int>();
        params->dataset_params.crop_corner_high =
            (params->dataset_params.crop_corner_high.cast<float>() * image_scale).cast<int>();
    }
    std::cout << "Scaled ROI " << params->dataset_params.crop_corner_low.transpose() << " "
              << params->dataset_params.crop_corner_high.transpose() << std::endl;
    std::cout << "Scaling input images by " << image_scale << std::endl;

    ct_params.Print();

    Vec3 voxel_size  = Vec3(ct_params.VoxelSizeX, ct_params.VoxelSizeY, ct_params.VoxelSizeZ);
    ivec3 voxels     = ivec3(ct_params.VoxelsX, ct_params.VoxelsY, ct_params.VoxelsZ);
    Vec3 volume_size = voxels.cast<double>().array() * voxel_size.cast<double>().array();
    volume_size *= params->dataset_params.volume_scale_factor;
    std::cout << "Volume Size Factor " << params->dataset_params.volume_scale_factor << std::endl;
    std::cout << "Volume Size " << volume_size.transpose() << std::endl;

    double max_volume_size = volume_size.array().maxCoeff();
    std::cout << "max_volume_size " << max_volume_size << std::endl;

    ivec2 detector_pixels     = ivec2(ct_params.DetectorPixelsX, ct_params.DetectorPixelsY);
    Vec2 detector_pixel_size  = Vec2(ct_params.DetectorPixelSizeX, ct_params.DetectorPixelSizeY);
    Vec2 actual_detector_size = detector_pixel_size.array() * detector_pixels.cast<double>().array();
    ivec2 cropped_size        = params->dataset_params.crop_corner_high - params->dataset_params.crop_corner_low;

    std::cout << "Detector Pixels " << detector_pixels.transpose() << std::endl;
    std::cout << "Cropped Pixels  " << cropped_size.transpose() << std::endl;
    std::cout << "Detector size " << actual_detector_size.transpose() << std::endl;

    crop_corner_low  = params->dataset_params.crop_corner_low;
    crop_corner_high = params->dataset_params.crop_corner_high;

    // After that the volume fits in the unit cube [-1, 1]
    scale_factor = 2. / max_volume_size;

    Vec2 detector_size = actual_detector_size * scale_factor;
    CameraBase cam;
    cam.K.cx = detector_pixels(0) / 2.f - 0.5 - crop_corner_low(0);
    cam.K.cy = detector_pixels(1) / 2.f - 0.5 - crop_corner_low(1);

    cam.K.fx = detector_pixels(0) / detector_size(0) * (ct_params.SrcToDetector * scale_factor);
    cam.K.fy = detector_pixels(1) / detector_size(1) * (ct_params.SrcToDetector * scale_factor);

    cam.w = cropped_size(0);
    cam.h = cropped_size(1);
    scene->cameras.push_back(cam);


    for (int i = 0; i < image_names.size(); ++i)
    {
        double ang = double(i) / (image_names.size() - 1) * 2 * pi<double>();
        auto img = std::make_shared<UnifiedImage>(ivec2(cropped_size(1), cropped_size(0)), 1);

        Vec3 dir             = Vec3(sin(-ang), -cos(-ang), 0);
        Vec3 source          = ct_params.SrcToObject * scale_factor * dir;
        Vec3 detector_center = -dir * (ct_params.SrcToDetector - ct_params.SrcToObject) * scale_factor;


        img->camera_id = 0;

        img->image_file = full_dir + "/" + image_names[i];

        Vec3 up      = Vec3(0, 0, -1);
        Vec3 forward = (detector_center - source).normalized();
        Vec3 right   = up.cross(forward).normalized();
        Mat3 R;
        R.col(0) = right;
        R.col(1) = up;
        R.col(2) = forward;

        img->pose.so3()         = Sophus::SO3d(R);
        img->pose.translation() = source;

        scene->frames.push_back(img);
    }

    if (std::filesystem::is_regular_file(params->train_params.split_index_file_train) &&
        std::filesystem::is_regular_file(params->train_params.split_index_file_test))
    {
        scene->train_indices = params->train_params.ReadIndexFile(params->train_params.split_index_file_train);
        scene->test_indices  = params->train_params.ReadIndexFile(params->train_params.split_index_file_test);
    }
    else
    {
        int num_total_images = scene->frames.size();
        std::vector<int> all_indices(num_total_images);
        std::iota(all_indices.begin(), all_indices.end(), 0);


        if (params->dataset_params.reduce_type == "uniform")
        {
            scene->train_indices = ReduceIndicesUniform(all_indices, params->train_params.max_images);
        }
        else if (params->dataset_params.reduce_type == "first")
        {
            scene->train_indices = all_indices;
            if (params->train_params.max_images > 0)
            {
                scene->train_indices.resize(params->train_params.max_images);
            }
        }
        else
        {
            CHECK(false);
        }
        scene->test_indices = scene->train_indices;
    }
#endif
}
void KaustSceneLoader::LoadAndPreprocessImages(TensorBoardLogger* logger)
{
#if 0
    auto ids = scene->train_indices;
    ids.insert(ids.end(), scene->test_indices.begin(), scene->test_indices.end());

    for (int image_id : ids)
    {
        CHECK_GE(image_id, 0);
        CHECK_LT(image_id, scene->frames.size());
        auto img = scene->frames[image_id];

        if (img->projection.defined()) continue;
        TemplatedImage<unsigned short> raw(img->image_file);

        TemplatedImage<float> converted(raw.dimensions());
        for (auto i : raw.rowRange())
        {
            for (auto j : raw.colRange())
            {
                converted(i, j) = raw(i, j);
            }
        }

        if (0)
        {
            ivec2 normalize_crop_start(50, 20);
            ivec2 normalize_crop_size(250, 250);
            auto normalize_crop = raw.getImageView().subImageView(normalize_crop_start(1), normalize_crop_start(0),
                                                                  normalize_crop_size(1), normalize_crop_size(0));
            std::vector<double> elems;
            for (int i : normalize_crop.rowRange())
            {
                for (int j : normalize_crop.colRange())
                {
                    elems.push_back(normalize_crop(i, j));
                }
            }
            float norm_factor = Statistics(elems).median;
            std::cout << "normalization factor " << norm_factor << std::endl;
            // std::cout << Statistics(elems) << std::endl;

            for (auto i : raw.rowRange())
            {
                for (auto j : raw.colRange())
                {
                    converted(i, j) = std::min(1.f, float(converted(i, j)) / norm_factor);
                }
            }
        }
        if (image_scale != 1)
        {
            TemplatedImage<float> scaled(converted.h * image_scale, converted.w * image_scale);
            converted.getImageView().copyScaleLinear(scaled.getImageView());
            converted = scaled;
        }

        if (logger)
        {
            auto raw_tensor = ImageViewToTensor(converted.getImageView());
            raw_tensor      = raw_tensor / raw_tensor.max();

            auto proj_gray = TensorToImage<unsigned char>(raw_tensor);
            TemplatedImage<ucvec3> raw_img(proj_gray.dimensions());
            ImageTransformation::Gray8ToRGB(proj_gray.getImageView(), raw_img.getImageView());
            if (scene->params->dataset_params.crop_corner_low(0) >= 0)
            {
                vec2 c1 = crop_corner_low.cast<float>();
                vec2 c2 = crop_corner_high.cast<float>();
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c1, vec2(c1(0), c2(1)), ucvec3(255, 0, 0));
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c1, vec2(c2(0), c1(1)), ucvec3(255, 0, 0));
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c2, vec2(c1(0), c2(1)), ucvec3(255, 0, 0));
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c2, vec2(c2(0), c1(1)), ucvec3(255, 0, 0));

#if 0
                c1 = normalize_crop_start.cast<float>() * image_scale;
                c2 = (normalize_crop_start + normalize_crop_size).cast<float>() * image_scale;
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c1, vec2(c1(0), c2(1)), ucvec3(0, 0, 255));
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c1, vec2(c2(0), c1(1)), ucvec3(0, 0, 255));
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c2, vec2(c1(0), c2(1)), ucvec3(0, 0, 255));
                ImageDraw::drawLineBresenham(raw_img.getImageView(), c2, vec2(c2(0), c1(1)), ucvec3(0, 0, 255));
#endif
            }
            LogImage(logger, raw_img, "Input/raw", image_id);
        }

        ivec2 cropped_size = crop_corner_high - crop_corner_low;

        TemplatedImage<float> cropped;
        cropped = converted.getImageView().subImageView(crop_corner_low(1), crop_corner_low(0), cropped_size(1),
                                                        cropped_size(0));



        img->projection = ImageViewToTensor(cropped.getImageView());
        PrintTensorInfo(img->projection);
    }

    // Process images
    float min_value = 98654965;
    float max_value = 0;
    for (auto img : scene->frames)
    {
        if (!img->projection.defined()) continue;
        float img_min = img->projection.min().item().toFloat();
        float img_max = img->projection.max().item().toFloat();
        std::cout << img->image_file << " img min/max " << img_min << "/" << img_max << std::endl;
        min_value = std::min(img_min, min_value);
        max_value = std::max(img_max, max_value);
    }
    std::cout << "Actual min/max: " << min_value << " " << max_value << std::endl;

    min_value = scene->params->dataset_params.xray_min;
    max_value = scene->params->dataset_params.xray_max;

    // min_value = 13046;
    // max_value = 65535;

    std::cout << "Using config min/max: " << min_value << " " << max_value << std::endl;

    // exit(0);
    for (int image_id = 0; image_id < scene->frames.size(); ++image_id)
    {
        auto img = scene->frames[image_id];
        if (!img->projection.defined()) continue;
        // convert transmittance to absorption
        img->projection = -torch::log10(img->projection / max_value);

        img->projection =
            img->projection * (scene->params->dataset_params.projection_factor / -std::log10(min_value / max_value));

        if (logger)
        {
            auto proj = TensorToImage<unsigned char>(img->projection);
            LogImage(logger, proj, "Input/processed", image_id);


            auto err_col_tens = ColorizeTensor(img->projection.squeeze(0) * 8, colorizeTurbo);
            auto proj_ampli   = TensorToImage<ucvec3>(err_col_tens);
            LogImage(logger, proj_ampli, "Input/amplified_x8", image_id);
        }

        PrintTensorInfo(img->projection);
    }
#endif
}
