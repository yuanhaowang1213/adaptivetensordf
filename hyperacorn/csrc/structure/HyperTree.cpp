#include "HyperTree.h"



// HyperTreeBaseImpl::HyperTreeBaseImpl(int d, int max_depth)
// {
//     int NS        = 1 << d;
//     int num_nodes = 0;
//     // NS^0 + NS^1 + NS^2 + ... + NS^max_depth
//     // NS^(depth+1)-1
//     for (int i = 0; i <= max_depth; ++i)
//     {
//         long n = 1;
//         for (int j = 0; j < i; ++j)
//         {
//             n *= NS;
//         }
//         num_nodes += n;
//     }

//     node_parent       = torch::empty({num_nodes}, torch::kInt32);
//     node_children     = -torch::ones({num_nodes, NS}, torch::kInt32);
//     node_position_min = torch::empty({num_nodes, d}, torch::kFloat32);
//     node_position_max = torch::empty({num_nodes, d}, torch::kFloat32);
//     node_scale        = torch::empty({num_nodes}, torch::kFloat32);
//     //    node_diagonal_length   = torch::empty({num_nodes}, torch::kFloat32);

//     // -1 means not computed yet
//     node_error             = -torch::ones({num_nodes}, torch::kFloat32);
//     node_max_density       = -torch::ones({num_nodes}, torch::kFloat32);

//     node_active            = torch::zeros({num_nodes}, torch::kInt32);
//     node_culled            = torch::zeros({num_nodes}, torch::kInt32);
//     node_depth             = torch::zeros({num_nodes}, torch::kInt32);
//     node_active_prefix_sum = torch::zeros({num_nodes}, torch::kInt32);

//     register_buffer("node_parent", node_parent);
//     register_buffer("node_children", node_children);
//     register_buffer("node_position_min", node_position_min);
//     register_buffer("node_position_max", node_position_max);
//     register_buffer("node_scale", node_scale);
//     register_buffer("node_error", node_error);
//     register_buffer("node_max_density", node_max_density);
//     register_buffer("node_active", node_active);
//     register_buffer("node_culled", node_culled);
//     register_buffer("node_depth", node_depth);
//     register_buffer("node_active_id", node_active_prefix_sum);

//     active_node_ids = torch::zeros({1}, torch::kLong);
//     SetActive(0);
//     register_buffer("active_node_ids", active_node_ids);

//     using Vec = Eigen::Matrix<float, -1, 1>;

//     auto get_node = [&](int node_id) -> std::pair<Vec, Vec>
//     {
//         Vec pos_min, pos_max;
//         pos_min.resize(d);
//         pos_max.resize(d);
//         for (int i = 0; i < d; ++i)
//         {
//             pos_min(i) = node_position_min.data_ptr<float>()[node_id * d + i];
//             pos_max(i) = node_position_max.data_ptr<float>()[node_id * d + i];
//         }
//         return {pos_min, pos_max};
//     };

//     auto set_node = [&](int node_id, int parent, Vec pos_min, Vec pos_max, int depth)
//     {
//         float scale = ((float)depth / (max_depth)) * 2 - 1;
//         if (max_depth == 0) scale = 0;

//         //        float diag = (pos_max - pos_min).norm();
//         for (int i = 0; i < d; ++i)
//         {
//             node_position_min.data_ptr<float>()[node_id * d + i] = pos_min(i);
//             node_position_max.data_ptr<float>()[node_id * d + i] = pos_max(i);
//         }
//         node_depth.data_ptr<int>()[node_id]   = depth;
//         node_parent.data_ptr<int>()[node_id]  = parent;
//         node_scale.data_ptr<float>()[node_id] = scale;
//         // node_diagonal_length.data_ptr<float>()[node_id] = diag;
//     };

//     int current_node = 0;
//     // nodes[current_node++] = Node(-1, 0, -Vec::Ones(), false, Vec::Ones());
// #if 1
//     int start_layer = 0;
//     set_node(current_node++, -1, -Vec(d).setOnes(), Vec(d).setOnes(), 0);
//     for (int depth = 1; depth <= max_depth; ++depth)
//     {
//         int n = current_node - start_layer;
//         for (int j = 0; j < n; ++j)
//         {
//             int node_id = start_layer + j;

//             auto [pos_min, pos_max] = get_node(node_id);
//             auto center             = (pos_min + pos_max) * 0.5;

//             for (int i = 0; i < NS; ++i)
//             {
//                 Vec new_min(d);
//                 Vec new_max(d);

//                 for (int k = 0; k < d; ++k)
//                 {
//                     if ((i >> k & 1) == 0)
//                     {
//                         new_min[k] = pos_min[k];
//                         new_max[k] = center[k];
//                     }
//                     else
//                     {
//                         new_min[k] = center[k];
//                         new_max[k] = pos_max[k];
//                     }
//                 }
//                 node_children.data_ptr<int>()[node_id * NS + i] = current_node;
//                 set_node(current_node++, node_id, new_min, new_max, depth);
//             }
//         }
//         start_layer = current_node - n * NS;
//     }
// #endif
//     for(int i = 0; i < num_nodes; ++i)
//     {
//         printf("test here 2\n");
//         auto [pos_min, pos_max] = get_node(i);
//         std::cout << "node i " << i << "pos " << pos_min << " " << pos_max << std::endl;
//     }
//     std::cout << "> Max Nodes2: " << num_nodes << std::endl;
// }
// To do using quad tree to represent the octree;
// HyperTreeBaseImpl::HyperTreeBaseImpl(int d, int max_depth)
// {
//     int NS;
//     bool use_quad = true;
//     float y_min = -0.3;
//     float y_max = 0.3;
//     if(use_quad)
//     {
//         NS              = 1 << (d-1);
//     }
//     else
//     {
//         NS              = 1 << d;
//     }
//     // int NS        = 1 << d;
//     int num_nodes = 0;
//     // NS^0 + NS^1 + NS^2 + ... + NS^max_depth
//     // NS^(depth+1)-1
//     for (int i = 0; i <= max_depth; ++i)
//     {
//         long n = 1;
//         for (int j = 0; j < i; ++j)
//         {
//             n *= NS;
//         }
//         num_nodes += n;
//     }

//     node_parent       = torch::empty({num_nodes}, torch::kInt32);
//     node_children     = -torch::ones({num_nodes, NS}, torch::kInt32);
//     node_position_min = torch::empty({num_nodes, d}, torch::kFloat32);
//     node_position_max = torch::empty({num_nodes, d}, torch::kFloat32);
//     node_scale        = torch::empty({num_nodes}, torch::kFloat32);
//     //    node_diagonal_length   = torch::empty({num_nodes}, torch::kFloat32);

//     // -1 means not computed yet
//     node_error             = -torch::ones({num_nodes}, torch::kFloat32);
//     node_max_density       = -torch::ones({num_nodes}, torch::kFloat32);

//     node_active            = torch::zeros({num_nodes}, torch::kInt32);
//     node_culled            = torch::zeros({num_nodes}, torch::kInt32);
//     node_depth             = torch::zeros({num_nodes}, torch::kInt32);
//     node_active_prefix_sum = torch::zeros({num_nodes}, torch::kInt32);

//     register_buffer("node_parent", node_parent);
//     register_buffer("node_children", node_children);
//     register_buffer("node_position_min", node_position_min);
//     register_buffer("node_position_max", node_position_max);
//     register_buffer("node_scale", node_scale);
//     register_buffer("node_error", node_error);
//     register_buffer("node_max_density", node_max_density);
//     register_buffer("node_active", node_active);
//     register_buffer("node_culled", node_culled);
//     register_buffer("node_depth", node_depth);
//     register_buffer("node_active_id", node_active_prefix_sum);

//     active_node_ids = torch::zeros({1}, torch::kLong);
//     SetActive(0);
//     register_buffer("active_node_ids", active_node_ids);

//     using Vec = Eigen::Matrix<float, -1, 1>;

//     auto get_node = [&](int node_id) -> std::pair<Vec, Vec>
//     {
//         Vec pos_min, pos_max;
//         pos_min.resize(d);
//         pos_max.resize(d);
//         for (int i = 0; i < d; ++i)
//         {
//             pos_min(i) = node_position_min.data_ptr<float>()[node_id * d + i];
//             pos_max(i) = node_position_max.data_ptr<float>()[node_id * d + i];
//         }
//         return {pos_min, pos_max};
//     };

//     // auto set_node = [&](int node_id, int parent, Vec pos_min, Vec pos_max, int depth)
//     // {
//     //     float scale = ((float)depth / (max_depth)) * 2 - 1;
//     //     if (max_depth == 0) scale = 0;

//     //     //        float diag = (pos_max - pos_min).norm();
//     //     for (int i = 0; i < d; ++i)
//     //     {
//     //         node_position_min.data_ptr<float>()[node_id * d + i] = pos_min(i);
//     //         node_position_max.data_ptr<float>()[node_id * d + i] = pos_max(i);
//     //     }
//     //     node_depth.data_ptr<int>()[node_id]   = depth;
//     //     node_parent.data_ptr<int>()[node_id]  = parent;
//     //     node_scale.data_ptr<float>()[node_id] = scale;
//     //     // node_diagonal_length.data_ptr<float>()[node_id] = diag;
//     // };
//     auto set_node = [&](int node_id, int parent, Vec pos_min, Vec pos_max, int depth, float y_min, float y_max)
//     {
//         float scale = ((float)depth/ (max_depth))*2 - 1;
//         if (max_depth == 0) scale = 0;

//         node_position_min.data_ptr<float>()[node_id * d + 0] = pos_min(0);
//         node_position_min.data_ptr<float>()[node_id * d + 1] = y_min;
//         node_position_min.data_ptr<float>()[node_id * d + 2] = pos_min(2);

//         node_position_max.data_ptr<float>()[node_id * d + 0] = pos_max(0);
//         node_position_max.data_ptr<float>()[node_id * d + 1] = y_max;
//         node_position_max.data_ptr<float>()[node_id * d + 2] = pos_max(2);
//         node_depth.data_ptr<int>()[node_id]   = depth;
//         node_parent.data_ptr<int>()[node_id]  = parent;
//         node_scale.data_ptr<float>()[node_id] = scale;
//     };

//     int current_node = 0;
//     // nodes[current_node++] = Node(-1, 0, -Vec::Ones(), false, Vec::Ones());
// #if 1
//     int start_layer = 0;
//     set_node(current_node++, -1, -Vec(d).setOnes(), Vec(d).setOnes(), 0, y_min, y_max);
//     for (int depth = 1; depth <= max_depth; ++depth)
//     {
//         int n = current_node - start_layer;
//         for (int j = 0; j < n; ++j)
//         {
//             int node_id = start_layer + j;

//             auto [pos_min, pos_max] = get_node(node_id);
//             auto center             = (pos_min + pos_max) * 0.5;

//             for (int i = 0; i < NS; ++i)
//             {
//                 Vec new_min(d);
//                 Vec new_max(d);

//                 for (int k = 0; k < d; ++k)
//                 {
//                     if ((i >> k & 1) == 0)
//                     {
//                         new_min[k] = pos_min[k];
//                         new_max[k] = center[k];
//                     }
//                     else
//                     {
//                         new_min[k] = center[k];
//                         new_max[k] = pos_max[k];
//                     }
//                 }
//                 node_children.data_ptr<int>()[node_id * NS + i] = current_node;
//                 set_node(current_node++, node_id, new_min, new_max, depth, y_min, y_max);
//             }
//         }
//         start_layer = current_node - n * NS;
//     }
// #endif
//     for(int i = 0; i < num_nodes; ++i)
//     {
//         printf("test here 2\n");
//         auto [pos_min, pos_max] = get_node(i);
//         std::cout << "node i " << i << "pos " << pos_min << " " << pos_max << std::endl;
//     }
//     std::cout << "> Max Nodes2: " << num_nodes << std::endl;
// }
// HyperTreeBaseImpl::HyperTreeBaseImpl(int d, int max_depth, bool use_quad, float y_min, float y_max)

HyperTreeBaseImpl::HyperTreeBaseImpl(int d_, int max_depth_)
{
    d = d_;
    max_depth =  max_depth_;
}

void HyperTreeBaseImpl::settree(bool use_quad, float y_min, float y_max)
{
    int NS;
    int num_nodes = 0;
    int temp_dim; 
    // bool use_quad = true;
    // float y_min = -0.3;
    // float y_max = 0.3;
    // NS^0 + NS^1 + NS^2 + ... + NS^max_depth
    // NS^(depth+1)-1

    // int d = depth[0];
    // int max_depth = depth[1];
    // bool use_quad;
    // if(depth[2] > 0)
    // {
    //     use_quad = true;
    // }
    // else
    // {
    //     use_quad = false;
    // }
    // float y_min = min_max[0];
    // float y_max = min_max[1];
    use_quad_tree = use_quad;
    std::cout << "use quad " << use_quad << std::endl;
    printf("y min %f %f\n", y_min, y_max);

    if(use_quad_tree )
    {
        NS              = 1 << (d-1);
        temp_dim        = d - 1;
    }
    else
    {
        NS              = 1 << d;
        temp_dim        = d;
    }
    for(int i = 0; i <= max_depth; ++i)
    {
        long n = 1;
        for(int j = 0; j < i; ++j)
        {
            n *= NS;
        }
        num_nodes += n;
    }
    // std::cout << "num nodes " << num_nodes << std::endl;
    node_parent             = torch::empty({num_nodes}, torch::kInt32);
    node_children           = -torch::ones({num_nodes, NS}, torch::kInt32 );
    node_position_min       = torch::empty({num_nodes, d}, torch::kFloat32);
    node_position_max       = torch::empty({num_nodes, d}, torch::kFloat32);
    node_scale              = torch::empty({num_nodes}, torch::kFloat32);

    node_error              = -torch::ones({num_nodes}, torch::kFloat32);
    node_max_density        = -torch::ones({num_nodes}, torch::kFloat32);

    node_active             = torch::zeros({num_nodes}, torch::kInt32);
    node_culled             = torch::zeros({num_nodes}, torch::kInt32);
    node_depth              = torch::zeros({num_nodes}, torch::kInt32);
    node_active_prefix_sum  = torch::zeros({num_nodes}, torch::kInt32);

    register_buffer("node_parent", node_parent);
    register_buffer("node_children", node_children);
    register_buffer("node_position_min", node_position_min);
    register_buffer("node_position_max", node_position_max);
    register_buffer("node_scale", node_scale);
    register_buffer("node_error", node_error);
    register_buffer("node_max_density", node_max_density);
    register_buffer("node_active", node_active);
    register_buffer("node_culled", node_culled);
    register_buffer("node_depth", node_depth);
    register_buffer("node_active_id", node_active_prefix_sum);

    active_node_ids = torch::zeros({1}, torch::kLong);
    SetActive(0);
    register_buffer("active_node_ids", active_node_ids);

    using Vec = Eigen::Matrix<float, -1, 1>;

    auto get_node = [&](int node_id) -> std::pair<Vec, Vec>
    {
        Vec pos_min, pos_max;
        pos_min.resize(d);
        pos_max.resize(d);
        for(int i = 0; i < d; ++i)
        {
            pos_min(i) = node_position_min.data_ptr<float>()[node_id * d +i];
            pos_max(i) = node_position_max.data_ptr<float>()[node_id * d +i];
        }
        return {pos_min, pos_max};
    };

    auto set_node_quad = [&](int node_id, int parent, Vec pos_min, Vec pos_max, int depth, float y_min, float y_max)
    {
        float scale = ((float)depth/ (max_depth))*2 - 1;
        if (max_depth == 0) scale = 0;

        node_position_min.data_ptr<float>()[node_id * d + 0] = pos_min(0);
        node_position_min.data_ptr<float>()[node_id * d + 1] = y_min;
        node_position_min.data_ptr<float>()[node_id * d + 2] = pos_min(2);

        node_position_max.data_ptr<float>()[node_id * d + 0] = pos_max(0);
        node_position_max.data_ptr<float>()[node_id * d + 1] = y_max;
        node_position_max.data_ptr<float>()[node_id * d + 2] = pos_max(2);
        node_depth.data_ptr<int>()[node_id]   = depth;
        node_parent.data_ptr<int>()[node_id]  = parent;
        node_scale.data_ptr<float>()[node_id] = scale;
    };

    auto set_node = [&](int node_id, int parent, Vec pos_min, Vec pos_max, int depth)
    {
        float scale = ((float)depth / (max_depth)) * 2 - 1;
        if (max_depth == 0) scale = 0;

        //        float diag = (pos_max - pos_min).norm();
        for (int i = 0; i < d; ++i)
        {
            node_position_min.data_ptr<float>()[node_id * d + i] = pos_min(i);
            node_position_max.data_ptr<float>()[node_id * d + i] = pos_max(i);
        }
        node_depth.data_ptr<int>()[node_id]   = depth;
        node_parent.data_ptr<int>()[node_id]  = parent;
        node_scale.data_ptr<float>()[node_id] = scale;
        // node_diagonal_length.data_ptr<float>()[node_id] = diag;
    };


    int current_node = 0;
#if 1
    int start_layer = 0;
    // std::cout << "max depth " << max_depth << std::endl;
    if(use_quad)
    {
        set_node_quad(current_node++, -1, -Vec(d).setOnes(), Vec(d).setOnes(), 0, y_min, y_max);

    }
    else
    {
        set_node(current_node++, -1, -Vec(d).setOnes(), Vec(d).setOnes(), 0);
    }
    // printf("test here\n");
    // for(int depth = 1; depth <= max_depth; ++depth)
    for(int depth = 1; depth <= max_depth; ++depth)
    {
        int n = current_node - start_layer;
        for(int j = 0; j < n; ++j)
        {
            int node_id = start_layer + j;
            auto [pos_min, pos_max] = get_node(node_id);
            auto center             = (pos_min + pos_max) * 0.5;
            // std::cout << "pos min " << pos_min[0] << " " << pos_min[1] << " " <<  pos_min[2] << std::endl;
            // std::cout << "pos max " << pos_max[0] << " " << pos_max[1] << " " <<  pos_max[2] << std::endl;

            // printf("test here 1\n");
            for(int i = 0; i < NS; ++i)
            {
                Vec new_min(d);
                Vec new_max(d);
                // for(int k = 0; k < d; ++k)
                for(int k = 0; k < temp_dim; ++k)
                {
                    int index;
                    if(temp_dim == 2 && k == 1)
                    {
                        index = 2;
                    }
                    else
                    {
                        index = k;
                    }
                    if ((i >> k & 1) == 0)
                    {
                        // new_min[k] = pos_min[k];
                        // new_max[k] = center[k];
                        new_min[index] = pos_min[index];
                        new_max[index] = center[index];
                    }
                    else
                    {
                        new_min[index] = center[index];
                        new_max[index] = pos_max[index];
                        // new_min[k] = center[k];
                        // new_max[k] = pos_max[k];                        
                    }
                }
                // std::cout << "new min " << new_min[0] << " " << new_min[1] << " " <<  new_min[2] << std::endl;
                // std::cout << "new max " << new_max[0] << " " << new_max[1] << " " <<  new_max[2] << std::endl;

                node_children.data_ptr<int>()[node_id * NS + i] = current_node;
                if(use_quad)
                {
                    set_node_quad(current_node++, node_id, new_min, new_max, depth, y_min, y_max);

                }
                else
                {
                    set_node(current_node++, node_id, new_min, new_max, depth);
                }

            }
        }
        start_layer = current_node - n * NS;

    }
#endif
    // for(int i = 0; i < num_nodes; ++i)
    // {
    //     int node_id = i;
    //     // auto [pos_min, pos_max] = get_node(i);
    //     // std::cout << "node i " << i << "pos " << pos_min[0] <<  pos_min[0]  << " " << pos_max << std::endl;
    //     // printf("node id %d pos %f %f %f %f %f %f\n", i, pos_min[0], pos_min[1], pos_min[2], pos_max[0], pos_max[1], pos_max[2]);
    //     printf("node id %d pos %f %f %f %f %f %f %f %f %f\n", i, node_position_min.data_ptr<float>()[node_id * d + 0], 
    //                                     node_position_min.data_ptr<float>()[node_id * d + 1], 
    //                                     node_position_min.data_ptr<float>()[node_id * d + 2], 
    //                                     node_position_max.data_ptr<float>()[node_id * d + 0], 
    //                                     node_position_max.data_ptr<float>()[node_id * d + 1], 
    //                                     node_position_max.data_ptr<float>()[node_id * d + 2],
    //                                     node_position_max.data_ptr<float>()[node_id * d ] - node_position_min.data_ptr<float>()[node_id * d ],
    //                                     node_position_max.data_ptr<float>()[node_id * d + 1] - node_position_min.data_ptr<float>()[node_id * d + 1],
    //                                     node_position_max.data_ptr<float>()[node_id * d + 2] - node_position_min.data_ptr<float>()[node_id * d + 2]
    //                                     );

    // }

    std::cout << "> Max Nodes: " << num_nodes << std::endl;
    // char c = getchar();
}

void HyperTreeBaseImpl::reset()
{
    register_buffer("node_parent", node_parent);
    register_buffer("node_children", node_children);
    register_buffer("node_position_min", node_position_min);
    register_buffer("node_position_max", node_position_max);
    register_buffer("node_scale", node_scale);
    register_buffer("node_error", node_error);
    register_buffer("node_active", node_active);
    register_buffer("node_culled", node_culled);
    register_buffer("node_depth", node_depth);
    register_buffer("node_active_id", node_active_prefix_sum);

    active_node_ids = torch::zeros({1}, torch::kLong);
    register_buffer("active_node_ids", active_node_ids);
}

void HyperTreeBaseImpl::SetActive(int depth)
{
    node_active.set_data((node_depth == depth).to(torch::kInt32));
    // PrintTensorInfo(node_active);
    UpdateActive();
}
void HyperTreeBaseImpl::UpdateActive()
{
    // A culled node is not allowed to be active
    CHECK_EQ((this->node_active.cpu() * this->node_culled.cpu()).sum().item().toInt(), 0);

    auto node_active = this->node_active.to(torch::kCPU);
    std::vector<long> active_node_ids;
    std::vector<long> node_active_prefix_sum;
    int active_count = 0;
    for (int i = 0; i < NumNodes(); ++i)
    {
        if (node_active.data_ptr<int>()[i] == 1)
        {
            active_node_ids.push_back(i);
            node_active_prefix_sum.push_back(active_count);
            active_count++;
        }
        else
        {
            node_active_prefix_sum.push_back(-1);
        }
    }

    std::cout << active_node_ids <<std::endl;
    // Use set_data because otherwise the registered buffer is broken.
    this->active_node_ids.set_data(torch::from_blob(&active_node_ids[0], {(long)active_node_ids.size()},
                                                    torch::TensorOptions().dtype(torch::kLong))
                                       .clone()
                                       .to(this->device()));
    this->node_active_prefix_sum.set_data(torch::from_blob(&node_active_prefix_sum[0],
                                                           {(long)node_active_prefix_sum.size()},
                                                           torch::TensorOptions().dtype(torch::kLong))
                                              .clone()
                                              .to(this->device()));


    // this->active_node_ids        = this->active_node_ids.to(node_position_min.device());
    // this->node_active_prefix_sum = this->node_active_prefix_sum.to(node_position_min.device());
}

torch::Tensor HyperTreeBaseImpl::InactiveNodeIds()
{
    auto node_active = this->node_active.to(torch::kCPU);
    std::vector<long> inactive_node_ids;
    for (int i = 0; i < NumNodes(); ++i)
    {
        if (node_active.data_ptr<int>()[i] == 0)
        {
            inactive_node_ids.push_back(i);
        }
    }
    return torch::from_blob(&inactive_node_ids[0], {(long)inactive_node_ids.size()},
                            torch::TensorOptions().dtype(torch::kLong))
        .clone()
        .to(this->device());
}

void HyperTreeBaseImpl::SetErrorForActiveNodes(torch::Tensor error, std::string strategy)
{
    CHECK_EQ(error.sizes(), node_error.sizes());
    CHECK_EQ(error.device(), node_error.device());

    auto active_float   = node_active.to(torch::kFloat32);
    auto inactive       = (1 - active_float);
    auto invalid_errors = (node_error < 0).to(torch::kFloat32);
    // Set all elements except the active to 0
    // error = active_float * error * node_max_density;
    error = active_float * error;

    // std::cout << "Updating Tree error with strategy = " << strategy << std::endl;
    // PrintTensorInfo(node_error);

    if (strategy == "override")
    {
        // Set all active elements to 0
        auto new_node_error = node_error * inactive;
        new_node_error      = new_node_error + error;
        node_error.set_data(new_node_error);
    }
    else if (strategy == "min")
    {
        // Set the inactive elements of the new erros to a large value
        // so min() does not use them
        error += inactive * 1000000;

        // Set the active elements of the old data to a large value,
        // that are still initialized with -1
        auto filtered_error = node_error + (invalid_errors * active_float) * 1000000;

        auto new_node_error = torch::min(filtered_error, error);
        node_error.set_data(new_node_error);
    }
    else
    {
        CHECK(false);
    }
    // PrintTensorInfo(node_error);
}

std::vector<AABB> HyperTreeBaseImpl::ActiveNodeBoxes()
{
    CHECK(active_node_ids.is_cpu());
    std::vector<AABB> result(NumActiveNodes());

    auto active_node_ids_ptr = active_node_ids.data_ptr<long>();
    auto box_min_ptr         = node_position_min.data_ptr<vec3>();
    auto box_max_ptr         = node_position_max.data_ptr<vec3>();

    for (int i = 0; i < result.size(); ++i)
    {
        auto nid  = active_node_ids_ptr[i];
        result[i] = AABB(box_min_ptr[nid], box_max_ptr[nid]);
    }
    return result;
}

void HyperTreeBaseImpl::UpdateCulling()
{
    auto culled   = node_culled.cpu();
    auto children = node_children.cpu();
    auto active   = node_active.cpu();

    int* culled_ptr   = culled.data_ptr<int>();
    int* children_ptr = children.data_ptr<int>();
    int* active_ptr   = active.data_ptr<int>();

    bool changed = true;

    while (changed)
    {
        changed = false;

        for (int i = 0; i < culled.size(0); ++i)
        {
            if (culled_ptr[i]) continue;

            bool all_culled = true;
            for (int cid = 0; cid < NS(); ++cid)
            {
                int c = children_ptr[i * NS() + cid];
                if (!culled_ptr[c]) all_culled = false;
            }
            if (all_culled)
            {
                CHECK(!active_ptr[i]);
                culled_ptr[i] = true;
                changed       = true;
            }
        }
    }

    node_culled.set_data(culled.to(node_culled.device()));

    // Set error of all culled nodes to 0
    node_error.mul_(1 - node_culled);
}
