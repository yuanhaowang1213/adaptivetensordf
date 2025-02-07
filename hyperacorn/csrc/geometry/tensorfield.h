#pragma once
#include <torch/script.h>

#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include "Settings.h"
class Tensor_PlaneQuadImpl : public torch::nn::Module
{
    public:
        Tensor_PlaneQuadImpl(int num_nodes, std::vector<std::vector<int>> matMode, std::vector<int> density_n_comp, 
                std::vector<long> feature_grid_shape, float init_scale = 0.5f, std::string init = "random" )
        {
            std::vector<std::string> plane_name = {"plane_coef_yz", "plane_coef_xz", "plane_coef_xy"};
            // float scale = 0.1;
            // float scale = 0.5f;
            scale = init_scale;
            int i = 0;
            int mat_id_0, mat_id_1;
            mat_id_0 = matMode[i][0];
            mat_id_1 = matMode[i][1];
            std::cout << "plane yz vector is " << mat_id_0 << "    " << mat_id_1 << std::endl;
            if(init == "random")
            {
                plane_coef_yz = scale * (torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[mat_id_1], feature_grid_shape[mat_id_0]}) -1.0f)*2.0f;
            }
            else if (init == "uniform")
            {
                plane_coef_yz = torch::empty({num_nodes, density_n_comp[i], feature_grid_shape[mat_id_1], feature_grid_shape[mat_id_0]});
                plane_coef_yz.uniform_(-scale,scale);
            }
            else
            {
                CHECK(false) << "Unknown grid init :"  << init << ". Expected: uniform, random";
            }
            i = 1;
            mat_id_0 = matMode[i][0];
            mat_id_1 = matMode[i][1];
            std::cout << "plane xz vector is " << mat_id_0 << "    " << mat_id_1 << std::endl;
            if(init == "random")
            {
                plane_coef_xz = scale * (torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[mat_id_1], feature_grid_shape[mat_id_0]}) -1.0f)*2.0f;
            }
            else if (init == "uniform")
            {
                plane_coef_xz = torch::empty({num_nodes, density_n_comp[i], feature_grid_shape[mat_id_1], feature_grid_shape[mat_id_0]});
                plane_coef_xz.uniform_(-scale,scale);
            }
            else
            {
                CHECK(false) << "Unknown grid init :"  << init << ". Expected: uniform, random";
            }
            i = 2;
            mat_id_0 = matMode[i][0];
            mat_id_1 = matMode[i][1];
            std::cout << "plane xy vector is " << mat_id_0 << "    " << mat_id_1 << std::endl;
            if(init == "random")
            {
                plane_coef_xy = scale * (torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[mat_id_1], feature_grid_shape[mat_id_0]})-1.0f)*2.0f;
            }
            else if (init == "uniform")
            {
                plane_coef_xy = torch::empty({num_nodes, density_n_comp[i], feature_grid_shape[mat_id_1], feature_grid_shape[mat_id_0]});
                plane_coef_xy.uniform_(-scale,scale);
            }
            else
            {
                CHECK(false) << "Unknown grid init :"  << init << ". Expected: uniform, random";
            }
            register_parameter("plane_coef_yz",plane_coef_yz);
            register_parameter("plane_coef_xz",plane_coef_xz);
            register_parameter("plane_coef_xy",plane_coef_xy);
            // grid_tmp = plane_coef_yz;
            plane_size_x = plane_coef_xz.sizes()[3];
            plane_size_y = plane_coef_yz.sizes()[3];
            plane_size_z = plane_coef_yz.sizes()[2];
            // PrintTensorInfo(plane_coef_yz);
            // printf("init here\n");
            // char c = getchar();
        }

        void reset()
        {
            torch::nn::init::uniform_(plane_coef_yz, -scale, scale);
            torch::nn::init::uniform_(plane_coef_xz, -scale, scale);
            torch::nn::init::uniform_(plane_coef_xy, -scale, scale);
        }
        at::Tensor forward( int index)
        {
            // CHECK(torch::equal(grid_tmp.sizes(), plane_coef_yz.sizes()));
            CHECK_EQ(plane_size_x, plane_coef_xy.sizes()[3]);
            CHECK_EQ(plane_size_y, plane_coef_xy.sizes()[2]);
            CHECK_EQ(plane_size_z, plane_coef_xz.sizes()[2]);
            if(index == 0)
            {
                // printf("choose plane yz\n");
                // return torch::index_select(plane_coef_yz, 0, node_index);
                return plane_coef_yz;
            }
            else if(index == 1)
            {
                // printf("choose plane xz\n");
                // return torch::index_select(plane_coef_xz, 0, node_index);
                return plane_coef_xz;
            }
            else if(index == 2)
            {
                // printf("choose plane xy\n");
                // return torch::index_select(plane_coef_xy,0, node_index);
                return plane_coef_xy;
            }
            else
            {
                CHECK(false);
                return torch::Tensor();
            }
        }
    torch::Tensor plane_coef_yz, plane_coef_xz, plane_coef_xy;
    float scale;
    // torch::Tensor grid_tmp;
    long plane_size_x;
    long plane_size_y;
    long plane_size_z;
};

TORCH_MODULE(Tensor_PlaneQuad);

// class Tensor_LineQuadImpl : public torch::nn::Module
// {
//     public:
//         Tensor_LineQuadImpl(int num_nodes, std::vector<int> vecMode, std::vector<int> density_n_comp, std::vector<long> feature_grid_shape, std::string init ="random")
//         {
//             std::vector<std::string> line_name = {"line_coef_x", "line_coef_y", "line_coef_z"};
//             float scale = 0.1;
//             int i = 0;
//             int vec_id;
//             vec_id = vecMode[i];
//             if(init == "random")
//             {
//                 line_coef_x = scale * torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[vec_id], 1});
//             }
//             else if(init == "uniform")
//             {
//                 line_coef_x = torch::empty({num_nodes,density_n_comp[i], feature_grid_shape[vec_id], 1});
//                 line_coef_x.uniform_(-1,1);
//             }
//             else
//             {
//                 CHECK(false) << "Unknown grid init: " << init << ". Expected: uniform, minus, zero";
//             }
//             std::cout << "line x vector is " << vec_id << "    "  << std::endl;

//             i = 1;
//             vec_id = vecMode[i];
//             if(init == "random")
//             {
//                 line_coef_y = scale * torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[vec_id], 1});
//             }
//             else if(init == "uniform")
//             {
//                 line_coef_y = torch::empty({num_nodes,density_n_comp[i], feature_grid_shape[vec_id], 1});
//                 line_coef_y.uniform_(-1,1);
//             }
//             else
//             {
//                 CHECK(false) << "Unknown grid init: " << init << ". Expected: uniform, minus, zero";
//             }
//             std::cout << "line y vector is " << vec_id << "    "  << std::endl;

//             i = 2;
//             vec_id = vecMode[i];
//             if(init == "random")
//             {
//                 line_coef_z = scale * torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[vec_id], 1});
//             }
//             else if(init == "uniform")
//             {
//                 line_coef_z = torch::empty({num_nodes,density_n_comp[i], feature_grid_shape[vec_id], 1});
//                 line_coef_z.uniform_(-1,1);
//             }
//             else
//             {
//                 CHECK(false) << "Unknown grid init: " << init << ". Expected: uniform, minus, zero";
//             }
//             std::cout << "line z vector is " << vec_id << "    "  << std::endl;
//             register_parameter("line_coef_x", line_coef_x);
//             register_parameter("line_coef_y", line_coef_y);
//             register_parameter("line_coef_z", line_coef_z);
//         }
//         at::Tensor forward(int index)
//         {
//             // CHECK(torch::equal(grid_tmp, plane_coef_yz));
//             if(index == 0)
//             {
//                 // return torch::index_select(line_coef_x, 0, node_index);
//                 return line_coef_x;
//             }
//             else if(index == 1)
//             {
//                 // return torch::index_select(line_coef_y, 0, node_index);
//                 return line_coef_y;
//             }
//             else if(index == 2)
//             {
//                 // return torch::index_select(line_coef_z,0, node_index);
//                 return line_coef_z;
//             }
//             else
//             {
//                 CHECK(false);
//                 return torch::Tensor();
//             }
//         }
//     torch::Tensor line_coef_x, line_coef_y, line_coef_z;
// };

class Tensor_LineQuadImpl : public torch::nn::Module
{
    public:
        Tensor_LineQuadImpl(int num_nodes, std::vector<int> vecMode, std::vector<int> density_n_comp,
                std::vector<long> feature_grid_shape, float init_scale = 0.5f, std::string init = "random")
        {
            std::vector<std::string> line_name = {"line_coef_x", "line_coef_y", "line_coef_z"};
            // float scale = 0.1;
            // float scale = 0.5f;
            scale = init_scale;
            int i = 0;
            int vec_id;
            vec_id = vecMode[i];
            if(init == "random")
            {
                line_coef_x = scale * (torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[vec_id], 1})-1.0f)*2.0f;
            }
            else if (init == "uniform")
            {
                line_coef_x =  torch::empty({num_nodes, density_n_comp[i], feature_grid_shape[vec_id], 1});
                line_coef_x.uniform_(-scale,scale);
            }
            else
            {
                CHECK(false) << "Unkown grid init: " << init << ". Expected: uniform, minus, zero" ;
            }
            std::cout << "line x vector is " << vec_id << "  " << std::endl;

            i = 1;
            vec_id = vecMode[i];
            if(init == "random")
            {
                line_coef_y = scale * (torch::rand({1, density_n_comp[i], feature_grid_shape[vec_id], 1})-1.0f)*2.0f;
            }
            else if (init == "uniform")
            {
                line_coef_y = torch::empty({1, density_n_comp[i], feature_grid_shape[vec_id], 1});
                line_coef_y.uniform_(-scale,scale);
            }
            else
            {
                CHECK(false) << "Unkown grid init: " << init << ". Expected: uniform, minus, zero" ;
            }
            std::cout << "line y vector is " << vec_id << "  " << std::endl;

            i = 2;
            vec_id = vecMode[i];
            if(init == "random")
            {
                line_coef_z = scale * (torch::rand({num_nodes, density_n_comp[i], feature_grid_shape[vec_id], 1})-1.0f)*2.0f;
            }
            else if (init == "uniform")
            {
                line_coef_z = torch::empty({num_nodes, density_n_comp[i], feature_grid_shape[vec_id], 1});
                line_coef_z.uniform_(-scale,scale);
            }
            else
            {
                CHECK(false) << "Unkown grid init: " << init << ". Expected: uniform, minus, zero" ;
            }
            std::cout << "line z vector is " << vec_id << "  " << std::endl;
            register_parameter("line_coef_x", line_coef_x);
            register_parameter("line_coef_y", line_coef_y);
            register_parameter("line_coef_z", line_coef_z);
        }

        void reset()
        {


            torch::nn::init::uniform_(line_coef_x, -scale, scale);
            torch::nn::init::uniform_(line_coef_y, -scale, scale);
            torch::nn::init::uniform_(line_coef_z, -scale, scale);

            // register_parameter("line_coef_x", line_coef_x);
            // register_parameter("line_coef_y", line_coef_y);
            // register_parameter("line_coef_z", line_coef_z);
            // line_coef_y.uniform_(-scale,scale);
            // line_coef_z.uniform_(-scale,scale);
            // register_parameter("line_coef_x", line_coef_x);
            // register_parameter("line_coef_y", line_coef_y);
            // register_parameter("line_coef_z", line_coef_z);
            // int count = 0;
            // for(auto& module : modules(/*include_self=*/false))
            // {
            //     count +=1;
            // }
            // printf("line module count is %d \n", count);
        }
        at::Tensor forward(int index)
        {
            // CHECK(torch::equal(grid_tmp, plane_coef_yz));
            if(index == 0)
            {
                // return torch::index_select(line_coef_x, 0, node_index);
                return line_coef_x;
            }
            else if(index == 1)
            {
                // return torch::index_select(line_coef_y, 0, node_index);
                return line_coef_y;
            }
            else if(index == 2)
            {
                // return torch::index_select(line_coef_z,0, node_index);
                return line_coef_z;
            }
            else
            {
                CHECK(false);
                return torch::Tensor();
            }
        }
    torch::Tensor line_coef_x, line_coef_y, line_coef_z;
    float scale;
};

TORCH_MODULE(Tensor_LineQuad);