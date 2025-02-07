#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "ImplicitNet.h"
#include "Settings.h"
#include "data/SceneBase.h"
#include "geometry.h"
#include "utils/cimg_wrapper.h"


#include "modules/IndirectGridSample2D.h"

#include "modules/IndirectGridSample1D.h"

#include "tensorfield.h"



class GeometryTensorQO : public HierarchicalNeuralGeometry
{
    public:
        GeometryTensorQO(int num_channels, int D, HyperTreeBase tree, std::shared_ptr<CombinedParams> params);
        virtual ~GeometryTensorQO() = default;
        torch::Tensor VolumeRegularizer();

        // virtual torch::Tensor Testcode() override;
    protected:
        void AddParametersToOptimizer() override;
        torch::Tensor SampleVolume(torch::Tensor global_coordinate, torch::Tensor node_id) override;
        virtual torch::Tensor SampleVolumeIndirect(torch::Tensor global_coordinate, torch::Tensor node_id) override;
        torch::Tensor SampleVolumeIndirect_feature(torch::Tensor global_coordinate, torch::Tensor node_id) ;

        std::tuple<torch::Tensor, torch::Tensor> SampleVolumeIndirect_edge(torch::Tensor global_coordinate, torch::Tensor node_id) ;
        // torch::Tensor SampleTV_feature(float scale, torch::Tensor plane_yz, torch::Tensor plane_xz, torch::Tensor plane_xy, 
                                                // torch::Tensor line_x, torch::Tensor line_y, torch::Tensor line_z, torch::Tensor in_roi_node);
        virtual void Compute_edge_nlm_samples(bool scale0_start) override;
        // torch::Tensor ComputeDensity(torch::Tensor global_coordinate);
        virtual torch::Tensor Testcode(std::string ep_dir) override;
        virtual void SaveTensor(std::string ep_dir) override;

    private:
        Tensor_LineQuad tensor_line_vector = nullptr;
        Tensor_PlaneQuad tensor_plane_vector = nullptr;

        ExplicitFeatureGrid explicit_grid_generator = nullptr;
        NeuralGridSampler grid_sampler              = nullptr;
        torch::Tensor nl_grid_tensor_yz, nl_grid_tensor_xz, nl_grid_tensor_xy;
        // torch::Tensor plane_yz_old, plane_xz_old, plane_xy_old;

        // torch::Tensor weight_yz, weight_xz, weight_xy;

        // VGG vggnet = nullptr;
        // Eigen::Vector<int, 3> nlm_shape_v_yz, nlm_shape_v_xz, nlm_shape_v_xy;

        // Tensor_Line tensor_line_vector = nullptr;
        // Tensor_Plane tensor_plane_vector = nullptr;

        // SampleList nlm_samples_yz, nlm_samples_xz, nlm_samples_xy;

        torch::Tensor fourier_grid, fourier_node_id;
        int fourier_size_x, fourier_size_y, fourier_size_z;

        torch::Tensor nlm_local_node_id;
        torch::Tensor nlm_node_id, nlm_xyz_grid;
        SampleList neighbor_samples;

        SampleList neighbor_samples_yz, neighbor_samples_xz, neighbor_samples_xy;

        torch::Tensor tv_line_x_coord, tv_line_y_coord,tv_line_z_coord ; //, tv_local_node_id, 
        
        torch::Tensor tv_line_x_coord_global, tv_line_z_coord_global;
        torch::Tensor tv_coord_global;
        torch::Tensor torch_plane_index;

        std::vector<float> compute_weight = {1.0f, 1.0f, 1.0f};
        // std::vector<std::vector<int>> matMode = {{0,1}, {0,2}, {1,2}};
        // std::vector<int> vecMode = {2,1,0};

        torch::Tensor in_roi_node_index, out_roi_node_index;
        std::vector<int> vec_grid_size = {30, 70, 30};
        std::vector<int> line_tv_index_z = {12,23};
        std::vector<std::vector<int>> matMode = {{1,2}, {0,2}, {0,1}};
        std::vector<int> vecMode = {0, 1, 2};

        torch::Tensor node_active_prefix_sum_inroi,node_active_prefix_sum_outroi;

        torch::Tensor node_active_prefix_sum;

};