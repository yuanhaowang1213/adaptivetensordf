/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"

#include "SceneBase.h"


template <int MODEL>
static __global__ void PointInAnyImage(StaticDeviceTensor<double, 2> point, StaticDeviceTensor<double, 2> rotation,
                                       StaticDeviceTensor<double, 2> translation,
                                       StaticDeviceTensor<double, 2> intrinsics, StaticDeviceTensor<float, 1> out_mask,
                                       vec3 roi_min, vec3 roi_max, int h, int w)
{
    int image_id = blockIdx.x;

    Quat q       = ((Quat*)&rotation(image_id, 0))[0];
    Vec3 t       = ((Vec3*)&translation(image_id, 0))[0];
    Vec5 k_coeff = ((Vec5*)&intrinsics(0, 0))[0];
    IntrinsicsPinholed K(k_coeff);
    K.s = 0;

    SE3 T = SE3(q, t).inverse();

    for (int point_id = threadIdx.x; point_id < point.sizes[0]; point_id += blockDim.x)
    {
        Vec3 position = ((Vec3*)&point(point_id, 0))[0];

        vec3 positionf = position.cast<float>();

        if (!BoxContainsPoint(roi_min.data(), roi_max.data(), positionf.data(), 3))
        {
            continue;
        }

        Vec3 view_pos = T * position;

        Vec2 image_pos;

        switch (MODEL)
        {
                // pinhole
            case 0:
                image_pos = K.project(view_pos);
                break;
                // orthographic
            case 1:
                image_pos = Vec2(K.fx * view_pos(0) + K.s * view_pos(1) + K.cx, K.fy * view_pos(1) + K.cy);
                break;
        }

        ivec2 pixel = image_pos.array().round().cast<int>();

        if (pixel(0) >= 0 && pixel(0) < w && pixel(1) >= 0 && pixel(1) < h)
        {
            out_mask(point_id) = 1;
        }
        // ((Vec3*)&out_point(tid, 0))[0] = new_pos;
    }
}


torch::Tensor SceneBase::PointInAnyImage(torch::Tensor points)
{
    auto linear_points = points.reshape({-1, 3}).to(torch::kDouble);

    auto mask = torch::zeros({linear_points.size(0)}, points.options().dtype(torch::kFloat32));

    if (dataset_params.camera_model == "pinhole")
    {
        ::PointInAnyImage<0><<<frames.size(), 128>>>(linear_points, pose->rotation, pose->translation,
                                                     camera_model->intrinsics, mask, dataset_params.roi_min,
                                                     dataset_params.roi_max, camera_model->h, camera_model->w);
    }
    else if (dataset_params.camera_model == "orthographic")
    {
        ::PointInAnyImage<1><<<frames.size(), 128>>>(linear_points, pose->rotation, pose->translation,
                                                     camera_model->intrinsics, mask, dataset_params.roi_min,
                                                     dataset_params.roi_max, camera_model->h, camera_model->w);
    }
    else
    {
        CHECK(false);
    }
    CUDA_SYNC_CHECK_ERROR();

    auto out_size = points.sizes().vec();
    out_size.pop_back();

    mask = mask.reshape(out_size);


    return mask;
}
