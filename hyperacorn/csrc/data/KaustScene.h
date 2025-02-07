/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/freeimage.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/time.h"
#include "saiga/core/util/ProgressBar.h"
#include "saiga/vision/torch/ImageTensor.h"

#include "SceneBase.h"
#include "Settings.h"

using namespace Saiga;


struct XTekCT : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(XTekCT);
    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override

    // SAIGA_PARAM_STRUCT(XTekCT);
    // SAIGA_PARAM_STRUCT_FUNCTIONS;
    // template <class ParamIterator>
    // void Params(ParamIterator* it) 
    {
        SAIGA_PARAM(VoxelsX);
        SAIGA_PARAM(VoxelsY);
        SAIGA_PARAM(VoxelsZ);
        SAIGA_PARAM(VoxelSizeX);
        SAIGA_PARAM(VoxelSizeY);
        SAIGA_PARAM(VoxelSizeZ);

        SAIGA_PARAM(DetectorPixelsX);
        SAIGA_PARAM(DetectorPixelsY);
        SAIGA_PARAM(DetectorPixelSizeX);
        SAIGA_PARAM(DetectorPixelSizeY);
        SAIGA_PARAM(SrcToObject);
        SAIGA_PARAM(SrcToDetector);
        SAIGA_PARAM(MaskRadius);
        SAIGA_PARAM(Projections);
    }

    void Print()
    {
        Table tab({30, 20});

        tab << "Name"
            << "Value";
        tab << "DetectorPixelsX" << DetectorPixelsX;
        tab << "DetectorPixelsY" << DetectorPixelsY;
        tab << "DetectorPixelSizeX" << DetectorPixelSizeX;
        tab << "DetectorPixelSizeY" << DetectorPixelSizeY;
        tab << "SrcToObject" << SrcToObject;
        tab << "SrcToDetector" << SrcToDetector;
        tab << "MaskRadius" << MaskRadius;
        tab << "Projections" << Projections;
        tab << "VoxelsX" << VoxelsX;
        tab << "VoxelsY" << VoxelsY;
        tab << "VoxelsZ" << VoxelsZ;
        tab << "VoxelSizeX" << VoxelSizeX;
        tab << "VoxelSizeY" << VoxelSizeY;
        tab << "VoxelSizeZ" << VoxelSizeZ;
    }
    // projection settings
    int DetectorPixelsX       = 0;
    int DetectorPixelsY       = 0;
    double DetectorPixelSizeX = 0;
    double DetectorPixelSizeY = 0;

    int VoxelsX = 0;
    int VoxelsY = 0;
    int VoxelsZ = 0;

    double VoxelSizeX = 0;
    double VoxelSizeY = 0;
    double VoxelSizeZ = 0;

    double SrcToObject   = 0;
    double SrcToDetector = 0;
    double MaskRadius    = 0;
    int Projections      = 0;
};

class KaustSceneLoader
{
   public:
    KaustSceneLoader(std::shared_ptr<CombinedParams> params);


    void LoadAndPreprocessImages(TensorBoardLogger* logger);

    // Set to 0.5 for half resolution
    double image_scale = 1.0;

    // This factor is multiplied to all geometry lengths
    double scale_factor;

    std::string dataset_name;
    XTekCT ct_params;

    ivec2 crop_corner_low;
    ivec2 crop_corner_high;

    std::shared_ptr<SceneBase> scene;
};