#define CUDA_NDEBUG

#include "saiga/cuda/cuda.h"
#include "saiga/vision/torch/CudaHelper.h"
#include "saiga/vision/torch/EigenTensor.h"

// #include "GpuBitonicSort.h"
#include "HyperTree.h"

#include "globalenv.h"
// #define in_roi(node_min, node_max, roi_min, roi_max) \
// node_mid = node_min + mode_max ; \
// ((node_min(0) > roi_min(0) &&  node_min(0) < roi_max(0)) || 
//         (node_max(0) > roi_min(0) && node_max(0) < roi_max(0)) ||
//         (node_mid(0) > roi_min(0) && node_mid(0) < roi_max(0))) &&
//         ((node_min(1) > roi_min(1) &&  node_min(1) < roi_max(1)) || 
//         (node_max(1) > roi_min(1) && node_max(1) < roi_max(1)) ||
//         (node_mid(1) > roi_min(1) && node_mid(1) < roi_max(1))) &&
//         ((node_min(2) > roi_min(2) &&  node_min(2) < roi_max(2)) || 
//         (node_max(2) > roi_min(2) && node_max(2) < roi_max(2)) ||
//         (node_mid(2) > roi_min(2) && node_mid(2) < roi_max(2))) ? (true) : (false)

template <int D>
struct DeviceHyperTree
{
    using Vec = Eigen::Vector<float, D>;
    StaticDeviceTensor<float, 2> node_position_min;
    StaticDeviceTensor<float, 2> node_position_max;
    StaticDeviceTensor<long, 1> active_node_ids;
    StaticDeviceTensor<int, 1> node_active;
    StaticDeviceTensor<int, 2> node_children;

    __host__ DeviceHyperTree(HyperTreeBaseImpl* tree)
    {
        node_position_min = tree->node_position_min;
        node_position_max = tree->node_position_max;
        active_node_ids   = tree->active_node_ids;
        node_active       = tree->node_active;
        node_children     = tree->node_children;
        //        node_diagonal_length = tree->node_diagonal_length;
        CHECK_EQ(node_position_min.strides[1], 1);
        CHECK_EQ(node_position_max.strides[1], 1);
    }

    __device__ Vec PositionMin(int node_id)
    {
        Vec* ptr = (Vec*)&node_position_min(node_id, 0);
        return ptr[0];
    }
    __device__ Vec PositionMax(int node_id)
    {
        Vec* ptr = (Vec*)&node_position_max(node_id, 0);
        return ptr[0];
    }
};

template <class T>
HD inline bool BoxContainsPointvec(T pos_min, T pos_max, T p, int D)
{
    for(int d = 0; d < D; ++d)
    {
        if (pos_min(d) > p(d) || pos_max(d) < p(d))
        {
            return false;
        }
    }
    return true;
}

template <int D>
static __global__ void ComputeLocalSamples(DeviceHyperTree<D> tree, StaticDeviceTensor<float, 3> global_samples,
                                           StaticDeviceTensor<long, 1> node_indices,
                                           StaticDeviceTensor<float, 3> out_local_samples)
{
    using Vec = typename DeviceHyperTree<D>::Vec;

    int group_id  = blockIdx.x;
    int local_tid = threadIdx.x;

    CUDA_KERNEL_ASSERT(group_id < global_samples.sizes[0]);
    int group_size = global_samples.sizes[1];
    int node_id    = node_indices(group_id);

    Vec pos_min = tree.PositionMin(node_id);
    Vec pos_max = tree.PositionMax(node_id);
    Vec size    = pos_max - pos_min;

    for (int i = local_tid; i < group_size; i += blockDim.x)
    {
        Vec c;
        for (int d = 0; d < D; ++d)
        {
            c(d) = global_samples(group_id, i, d);
        }
        c = c - pos_min;
        // [0, 1]
        c = c.array() / size.array();
        // [-1, 1]
        c = (c * 2) - Vec::Ones();

        ((Vec*)&out_local_samples(group_id, i, 0))[0] = c;
    }
}


template <int D>
static __global__ void ComputeLocalSamples2(DeviceHyperTree<D> tree, StaticDeviceTensor<float, 2> global_samples,
                                            StaticDeviceTensor<long, 1> node_indices,
                                            StaticDeviceTensor<float, 2> out_local_samples)
{
    using Vec = typename DeviceHyperTree<D>::Vec;

    int sample_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (sample_id >= global_samples.sizes[0]) return;

    int node_id = node_indices(sample_id);

    Vec pos_min = tree.PositionMin(node_id);
    Vec pos_max = tree.PositionMax(node_id);
    Vec size    = pos_max - pos_min;


    Vec c;
    for (int d = 0; d < D; ++d)
    {
        c(d) = global_samples(sample_id, d);
    }
    c = c - pos_min;
    // [0, 1]
    c = c.array() / size.array();
    // [-1, 1]
    c = (c * 2) - Vec::Ones();

    ((Vec*)&out_local_samples(sample_id, 0))[0] = c;
}
// torch::Tensor HyperTreeBaseImpl::ComputeLocalSamples(torch::Tensor global_samples, torch::Tensor node_indices)
// {
//     CHECK(global_samples.is_cuda());
//     CHECK(node_position_min.is_cuda());


//     auto local_samples = torch::zeros_like(global_samples);

//     if (global_samples.dim() == 3 && node_indices.dim() == 1)
//     {
// #if 0
//         CHECK_EQ(global_samples.dim(), 3);
//         if (global_samples.size(0) > 0)
//         {
//             switch (D())
//             {
//                 case 3:
//                     ::ComputeLocalSamples<3>
//                         <<<global_samples.size(0), 128>>>(this, global_samples, node_indices, local_samples);
//                     break;
//                 default:
//                     CHECK(false);
//             }
//         }
// #endif


//         auto sample_pos_min = torch::index_select(node_position_min, 0, node_indices).unsqueeze(1);
//         auto sample_pos_max = torch::index_select(node_position_max, 0, node_indices).unsqueeze(1);
//         auto size           = sample_pos_max - sample_pos_min;

//         //    PrintTensorInfo(sample_pos_min);
//         //    PrintTensorInfo(size);

//         auto local_samples2 = (global_samples - sample_pos_min) / size * 2 - 1;

//         //    PrintTensorInfo(local_samples);
//             // printf("local samples should be\n");
//             // PrintTensorInfo(sample_pos_min);
//             // PrintTensorInfo(local_samples2);
//             // PrintTensorInfo(size);
//             // PrintTensorInfo(global_samples);
//             // auto global_samples2 = global_samples.to(torch::kCPU);
//             // auto size2 = size.to(torch::kCPU);
//             // auto local_samples22 = local_samples2.to(torch::kCPU);
//             // auto sample_pos_min2 = sample_pos_min.to(torch::kCPU);
//             // std::cout << "global_samples " << global_samples2.data_ptr<float>()[0 ] <<" "<< global_samples2.data_ptr<float>()[global_samples2.stride(2)] <<" "<< global_samples2.data_ptr<float>()[2*global_samples2.stride(2)] << std::endl;
//             // std::cout << "sample_pos_min " << sample_pos_min2.data_ptr<float>()[0 ]<<" " << sample_pos_min2.data_ptr<float>()[sample_pos_min2.stride(2)] <<" "<< sample_pos_min2.data_ptr<float>()[2*sample_pos_min2.stride(2)] << std::endl;
//             // std::cout << "size " << size2.data_ptr<float>()[0 ]<<" " << size2.data_ptr<float>()[size2.stride(2)] <<" "<< size2.data_ptr<float>()[2*size2.stride(2)] << std::endl;


//             // std::cout << "local_samples2 " << local_samples22.data_ptr<float>()[0 ]<<" " << local_samples22.data_ptr<float>()[local_samples22.stride(2)]<<" " << local_samples22.data_ptr<float>()[2*local_samples22.stride(2)] << std::endl;

//         //    PrintTensorInfo(local_samples - local_samples2);

//         CUDA_SYNC_CHECK_ERROR();
//         return local_samples2;
//     }
//     else
//     {
// #if 0
//         if (global_samples.size(0) > 0)
//         {
//             switch (D())
//             {
//                 case 3:
//                     ::ComputeLocalSamples2<3>
//                         <<<global_samples.size(0), 128>>>(this, global_samples, node_indices, local_samples);
//                     break;
//                 default:
//                     CHECK(false);
//             }
//         }
// #endif

//         // printf("test here 2\n");
//         auto sample_pos_min = torch::index_select(node_position_min, 0, node_indices);
//         auto sample_pos_max = torch::index_select(node_position_max, 0, node_indices);
//         auto size           = sample_pos_max - sample_pos_min;

//         PrintTensorInfo(sample_pos_min);
//         PrintTensorInfo(size);
//         PrintTensorInfo(global_samples);

//         auto local_samples2 = (global_samples - sample_pos_min) / size * 2 - 1;

//         // PrintTensorInfo(local_samples);
//         // PrintTensorInfo(local_samples2);
//         // PrintTensorInfo(local_samples - local_samples2);
//         // exit(0);

//         CUDA_SYNC_CHECK_ERROR();
//         return local_samples2;
//     }

//     // PrintTensorInfo(node_position_min);
//     // PrintTensorInfo(node_position_max);
//     // PrintTensorInfo(global_samples);
//     // PrintTensorInfo(node_indices);
// }

torch::Tensor HyperTreeBaseImpl::ComputeLocalSamples(torch::Tensor global_samples, torch::Tensor node_indices)
{
    CHECK(global_samples.is_cuda());
    CHECK(node_position_min.is_cuda());
    torch::Tensor sample_pos_min, sample_pos_max;
    if (global_samples.dim() == 3 && node_indices.dim() == 1)
    {
        sample_pos_min = torch::index_select(node_position_min, 0, node_indices).unsqueeze(1).contiguous();
        sample_pos_max = torch::index_select(node_position_max, 0, node_indices).unsqueeze(1).contiguous();
    }
    else
    {
        sample_pos_min = torch::index_select(node_position_min, 0, node_indices).contiguous();
        sample_pos_max = torch::index_select(node_position_max, 0, node_indices).contiguous();
    }
    auto size           = sample_pos_max - sample_pos_min;

    auto local_samples2 = ((global_samples - sample_pos_min) / size * 2 - 1).contiguous();

    CUDA_SYNC_CHECK_ERROR();
    return local_samples2;

}

struct CompareInterval
{
    HD inline bool operator()(float kA, int vA, float kB, int vB) const { return kA < kB; }
};

template <int D, int ThreadsPerBlock>
static __global__ void ComputeRaySamples(
    DeviceHyperTree<D> tree, StaticDeviceTensor<float, 2> ray_origin, StaticDeviceTensor<float, 2> ray_direction,
    float* sample_rnd, int* out_num_samples,
    StaticDeviceTensor<float, 2> out_global_coordinates, StaticDeviceTensor<float, 1> out_weight,
    StaticDeviceTensor<float, 1> out_ray_t, StaticDeviceTensor<long, 1> out_ray_index,
    StaticDeviceTensor<long, 1> out_ray_local_id, StaticDeviceTensor<long, 1> out_node_id, int max_samples_per_node)
{
    using Vec     = typename DeviceHyperTree<D>::Vec;
    int ray_id    = blockIdx.x;
    int lane_id   = threadIdx.x % 32;
    int warp_id   = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    constexpr int max_intersections = 256;
    __shared__ int num_intersections;
    __shared__ int num_samples_of_ray;
    __shared__ int global_sample_offset;
    __shared__ int inter_num_samples[max_intersections];
    __shared__ int inter_num_samples_scan[max_intersections];
    __shared__ float inter_tmin[max_intersections];
    __shared__ float inter_tmax[max_intersections];
    __shared__ int inter_node_id[max_intersections];
    __shared__ int inter_id[max_intersections];

    if (threadIdx.x == 0)
    {
        num_intersections    = 0;
        num_samples_of_ray   = 0;
        global_sample_offset = 0;
    }

    for (int i = threadIdx.x; i < max_intersections; i += blockDim.x)
    {
        // Tmin is used for sorting therefore we need to set it far away
        inter_tmin[i]        = 12345678;
        inter_num_samples[i] = 0;
        inter_id[i]          = i;
    }

    __syncthreads();

    // Each block processes one ray
    Vec origin;
    Vec direction;
    for (int d = 0; d < D; ++d)
    {
        origin(d)    = ray_origin(ray_id, d);
        direction(d) = ray_direction(ray_id, d);
    }

    int num_active_nodes = tree.active_node_ids.sizes[0];

    // Use the complete block to test all active nodes
    for (int i = threadIdx.x; i < num_active_nodes; i += blockDim.x)
    {
        int node_id = tree.active_node_ids(i);

        Vec box_min, box_max;
        for (int d = 0; d < D; ++d)
        {
            box_min(d) = tree.node_position_min(node_id, d);
            box_max(d) = tree.node_position_max(node_id, d);
        }

        auto [hit, tmin, tmax] =
            IntersectBoxRayPrecise(box_min.data(), box_max.data(), origin.data(), direction.data(), D);

        if (tmax - tmin < 1e-5)
        {
            continue;
        }

        if (hit)
        {
            auto index = atomicAdd(&num_intersections, 1);
            CUDA_KERNEL_ASSERT(index < max_intersections);
            inter_tmin[index]    = tmin;
            inter_tmax[index]    = tmax;
            inter_node_id[index] = node_id;

            float diag               = (box_max - box_min).norm();
            // printf("hit %f ori %f %f %f dir %f %f %f box %f %f %f %f %f %f node %f %f %d %d\n", diag, origin(0), origin(1),
            //                     origin(2), direction(0), direction(1), direction(2), 
            //                     box_min(0), box_min(1), box_min(2), box_max(0), box_max(1), box_max(2), 
            //                     tmin, tmax, node_id, max_samples_per_node);
            float dis                = tmax - tmin;
            float rel_dis            = dis / diag;
            int num_samples          = iCeil(rel_dis * max_samples_per_node);
            inter_num_samples[index] = num_samples;

            atomicAdd(&num_samples_of_ray, num_samples);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0)
    {
        global_sample_offset = atomicAdd(out_num_samples, num_samples_of_ray);
        atomicMax(out_num_samples + 1, num_samples_of_ray);
        // printf("global sample offset %d %d %d\n", out_num_samples[0] , num_samples_of_ray, out_weight.sizes[0]);
        // printf("num samples %d %d %d \n",global_sample_offset , num_samples_of_ray,  out_weight.sizes[0]);

        CUDA_KERNEL_ASSERT(global_sample_offset + num_samples_of_ray < out_weight.sizes[0]);
    }

    // caffe2::bitonicSort<CompareInterval, float, int, max_intersections, ThreadsPerBlock>(inter_tmin, inter_id,
    //                                                                                      CompareInterval());

    // TODO: more efficient scan
    for (int i = threadIdx.x; i < max_intersections; i += blockDim.x)
    {
        int count = 0;
        for (int j = 0; j < i; ++j)
        {
            int sorted_id   = inter_id[j];
            int num_samples = inter_num_samples[sorted_id];
            count += num_samples;
        }
        inter_num_samples_scan[i] = count;
    }

    __syncthreads();

#if 0
    for (int i = threadIdx.x; i < num_intersections - 1; i += blockDim.x)
    {
        float2 interval_this = {inter_tmin[i], inter_tmax[inter_id[i]]};
        float2 interval_next = {inter_tmin[i + 1], inter_tmax[inter_id[i + 1]]};
        // CUDA_KERNEL_ASSERT(std::abs(interval_this.y - interval_next.x) <= 1e-4);

        CUDA_KERNEL_ASSERT(intersection_rnd == nullptr);

        if (intersection_rnd)
        {
            float rnd         = intersection_rnd[global_sample_offset + i];
            float max_overlap = 0.05;

            float center_min = interval_this.x * max_overlap + interval_this.y * (1 - max_overlap);
            float center_max = interval_next.x * (1 - max_overlap) + interval_next.y * max_overlap;

            // float center_t = center_min * rnd + center_max * (1 - rnd);
            float center_t;
            if (rnd < 0.5)
            {
                rnd      = rnd * 2;
                center_t = center_min * rnd + interval_this.y * (1 - rnd);
            }
            else
            {
                rnd      = (rnd - 0.5) * 2;
                center_t = center_max * rnd + interval_next.x * (1 - rnd);
            }

            inter_tmax[inter_id[i]] = center_t;
            inter_tmin[i + 1]       = center_t;

            float2 interval_this = {inter_tmin[i], inter_tmax[inter_id[i]]};
            float2 interval_next = {inter_tmin[i + 1], inter_tmax[inter_id[i + 1]]};
            CUDA_KERNEL_ASSERT(std::abs(interval_this.y - interval_next.x) <= 1e-4);
        }
    }
#endif

    __syncthreads();

    // Process each intersection by a single warp
    for (int iid = warp_id; iid < num_intersections; iid += num_warps)
    {
        // the id and tmin was sorted therefore we can use iid
        int sorted_id        = inter_id[iid];
        float tmin           = inter_tmin[iid];
        int num_samples_scan = inter_num_samples_scan[iid];
        // these values have not been sorted -> use the sorted index to access them
        int node_id     = inter_node_id[sorted_id];
        float tmax      = inter_tmax[sorted_id];
        int num_samples = inter_num_samples[sorted_id];


        CUDA_KERNEL_ASSERT(num_samples <= max_samples_per_node);
        if (num_samples == 0) continue;

        float dis    = tmax - tmin;
        float weight = dis / num_samples;
        float step   = (tmax - tmin) / num_samples;

        // Vec box_min, box_max;
        // for(int d = 0; d < D; ++d)
        // {
        //     box_min(d) = tree.node_position_min(node_id, d);
        //     box_max(d) = tree.node_position_max(node_id, d);
        // }

        int out_sample_index = 0;
        out_sample_index = global_sample_offset + num_samples_scan;

        for (int j = lane_id; j < num_samples; j += 32)
        {
            int global_sample_idx = out_sample_index + j;
            float t1              = tmin + j * step;
            float t2              = tmin + (j + 1) * step;

            float a = 0.5;
            if (sample_rnd)
            {
                a = sample_rnd[global_sample_idx];
            }
            float t        = t1 * (1 - a) + t2 * a;
            Vec global_pos = origin + t * direction;
            // if(!BoxContainsPointvec<Vec>(box_min, box_max, global_pos, D))
            // {
            //     printf("out side of the node min %f %f %f max %f %f %f global %f %f %f\n",box_min(0),box_min(1),box_min(2),
            //                                                     box_max(0), box_max(1), box_max(2),
            //                                                     global_pos(0),global_pos(1),global_pos(2));
            // }

            for (int d = 0; d < D; ++d)
            {
                out_global_coordinates(global_sample_idx, d) = global_pos(d);
            }
            out_ray_t(global_sample_idx)        = t;
            out_weight(global_sample_idx)       = weight;
            out_node_id(global_sample_idx)      = node_id;
            out_ray_index(global_sample_idx)    = ray_id;
            out_ray_local_id(global_sample_idx) = num_samples_scan + j;
        }

        __syncwarp();
        for (int j = lane_id; j < num_samples; j += 32)
        {
            float t = out_ray_t(out_sample_index + j);

            float t_half_min;
            if (j == 0)
                t_half_min = tmin;
            else
                t_half_min = (out_ray_t(out_sample_index + j - 1) + t) * 0.5f;

            float t_half_max;
            if (j == num_samples - 1)
                t_half_max = tmax;
            else
                t_half_max = (out_ray_t(out_sample_index + j + 1) + t) * 0.5f;

            out_weight(out_sample_index + j) = t_half_max - t_half_min;
        }
    }
}

SampleList HyperTreeBaseImpl::CreateSamplesForRays(const RayList& rays, int max_samples_per_node, bool interval_jitter)
{
    CHECK(rays.direction.is_cuda());
    CHECK(node_position_min.is_cuda());

    int predicted_samples = iCeil(rays.size() * max_samples_per_node * pow(NumActiveNodes(), 1.0 / D()));
    // int predicted_samples = iCeil(rays.size() * max_samples_per_node * pow(NumActiveNodes(), 1.0 ));

    // std::cout << "ray sampling para " << rays.size() << " " << max_samples_per_node <<" " << pow(NumActiveNodes(), 1.0 / D()) << std::endl;

    // std::cout << predicted_samples << std::endl;
    // std::cout << NumActiveNodes() << " " <<  1.0 / D() << std::endl;
    SampleList list;
    list.Allocate(predicted_samples, D(), node_position_min.device());

    // std::cout << "node position " << TensorInfo(node_position_min) << " prediced samples " << predicted_samples<< std::endl;

    torch::Tensor interval_rnd = interval_jitter ? torch::rand_like(list.weight) : torch::Tensor();
    float* interval_rnd_ptr    = interval_jitter ? interval_rnd.data_ptr<float>() : nullptr;

    auto out_num_samples_max_per_ray = torch::zeros({2}, node_position_min.options().dtype(torch::kInt32));

    switch (D())
    {
        case 3:
            ::ComputeRaySamples<3, 128><<<rays.size(), 128>>>(
                this, rays.origin, rays.direction, interval_rnd_ptr,
                out_num_samples_max_per_ray.data_ptr<int>(), list.global_coordinate, list.weight, list.ray_t,
                list.ray_index, list.local_index_in_ray, list.node_id, max_samples_per_node);
            break;
        default:
            CHECK(false);
    }

    out_num_samples_max_per_ray = out_num_samples_max_per_ray.cpu();
    int actual_samples          = out_num_samples_max_per_ray.data_ptr<int>()[0];
    list.max_samples_per_ray    = out_num_samples_max_per_ray.data_ptr<int>()[1];
    list.Shrink(actual_samples);
    CHECK_LE(actual_samples, predicted_samples);

    {
        // Use this method for position computation to get the correct positional gradient
        auto origin2           = torch::index_select(rays.origin, 0, list.ray_index);
        auto dir2              = torch::index_select(rays.direction, 0, list.ray_index);
        auto pos2              = origin2 + dir2 * list.ray_t.unsqueeze(1);

        // std::cout << "pos 2" << TensorInfo(rays.origin)<< " " << TensorInfo(rays.direction) << " " << TensorInfo(list.ray_index) << "predicted sampled " << predicted_samples << std::endl;

        // auto nodeselectmin     = torch::index_select(node_position_min, 0, list.node_id);
        // auto nodeselectmax     = torch::index_select(node_position_max, 0, list.node_id); 
        
        // auto bandtestmin       = pos2 - nodeselectmin;
        // auto bandtestmax       = nodeselectmax - pos2;
        // // auto bandtestmin          = torch::gt(pos2 - nodeselectmin,0) ;
        // // auto bandtestmax        = torch::gt(node_position_max-pos2, 0);
        // // auto bandtest           = bandtestmin | bandtestmax;
        // float min1 = bandtestmin.min().item().toFloat();
        // float min2 = bandtestmax.min().item().toFloat();
        // // printf("node information %f %f\n", min1, min2);
        // // PrintTensorInfo(pos2);
        // // PrintTensorInfo(list.node_id);
        // // PrintTensorInfo(node_position_min);
        // // PrintTensorInfo(node_position_max);
        // // PrintTensorInfo(nodeselectmin);
        // // PrintTensorInfo(nodeselectmax);
        // if(bandtestmin.min().item().toFloat() < 0 || bandtestmax.min().item().toFloat() < 0 )
        // {
        //     printf("there is an error %f %f\n",min1, min2);
        // }
        
        // PrintTensorInfo(bandtestmin);
        // PrintTensorInfo(bandtestmax);
        // CHECK_GT(bandtestmin,0);
        // CHECK_GT(bandtestmax,0);
        // PrintTensorInfo(bandtest);
        // printf("global pos sample ray\n");
        // PrintTensorInfo(pos2);
        // // std::cout << pos2.sizes() << std::endl;

        // std::cout << "x dim info" << std::endl;
        // PrintTensorInfo(pos2.slice(1,0,1));
        // std::cout << "y dim info" << std::endl;
        // PrintTensorInfo(pos2.slice(1,1,2));
        // std::cout << "z dim info" << std::endl;
        // PrintTensorInfo(pos2.slice(1,2,3));


        list.global_coordinate = pos2;
    }

    // std::cout << "> CreateSamplesForRays: " << actual_samples << "/" << predicted_samples
    //           << " Max Per Ray: " << list.max_samples_per_ray << std::endl;
    CUDA_SYNC_CHECK_ERROR();
    return list;
}

template <int D> 
static __global__ void RandomGlobalSamples(DeviceHyperTree<D> tree, StaticDeviceTensor<int, 1> node_ids, 
                                            ivec3 grid_size, StaticDeviceTensor<float, 5> rand_value, StaticDeviceTensor<float, 5> out_position)
{
    CUDA_KERNEL_ASSERT(blockIdx.x < node_ids.sizes[0]);
    int node_id = node_ids((int)blockIdx.x);
    int thread_id = threadIdx.x;

    vec3 pos_min = tree.PositionMin(node_id);
    vec3 pos_max = tree.PositionMax(node_id);
    vec3 size    = pos_max - pos_min;

    int total_cells = grid_size.array().prod();
    for(int cell_id = thread_id; cell_id < total_cells; cell_id += blockDim.x)
    {
        ivec3 res;
        ivec3 localpos;
        int tmp = cell_id;
        for(int d = 0; d < D; ++d)
        {
            res(d) = tmp % grid_size(d);
            tmp /= grid_size(d);
        }
        // for(int d = 0; d < D ; ++d)
        // {
        //     localpos(d) = rand_value((int)blockIdx.x, res(2), res(1), res(0), d);
        // }
        // vec3 local_position = localpos.cast<float>().array() ;
        // vec3 global_position = local_position.array() * size.array();
        // global_position += pos_min;
        for(int d = 0; d < D; ++d)
        {
            // out_position((int)blockIdx.x, res(2), res(1), res(0), d) = local_position(d);
            out_position((int)blockIdx.x, res(2), res(1), res(0), d) = rand_value((int)blockIdx.x, res(2), res(1), res(0), d) * size(d) + pos_min(d);

        }
    }
}

template <int D, int ThreadsPerBlock> 
static __global__ void ComputeRaySamples_area(
    DeviceHyperTree<D> tree, StaticDeviceTensor<float, 2> ray_origin, StaticDeviceTensor<float, 2> ray_direction,
    float * sample_rnd, int * out_num_samples,
    StaticDeviceTensor<float, 2> out_global_coordinates, StaticDeviceTensor<float, 1> out_weight,
    StaticDeviceTensor<float, 1> out_ray_t, StaticDeviceTensor<long, 1> out_ray_index,
    StaticDeviceTensor<long, 1> out_ray_local_id, StaticDeviceTensor<long, 1> out_node_id, int max_samples_per_node,
    vec3 tree_roi_min, vec3 tree_roi_max)
{
    using Vec     = typename DeviceHyperTree<D>::Vec;
    int ray_id    = blockIdx.x;
    int lane_id   = threadIdx.x % 32;
    int warp_id   = threadIdx.x / 32;
    int num_warps = blockDim.x / 32;

    constexpr  int  max_intersections = 256;
    __shared__ int  num_intersections;
    __shared__ int  num_samples_of_ray;
    __shared__ int  global_sample_offset;
    __shared__ int  inter_num_samples[max_intersections];
    __shared__ int  inter_num_samples_scan[max_intersections];
    __shared__ float inter_tmin[max_intersections];
    __shared__ float inter_tmax[max_intersections];
    __shared__ int   inter_node_id[max_intersections];
    __shared__ int   inter_id[max_intersections];

    if(threadIdx.x == 0)
    {
        num_intersections       = 0;
        num_samples_of_ray       = 0;
        global_sample_offset    = 0;
    } 
    for(int i = threadIdx.x; i < max_intersections; i+= blockDim.x)
    {
        // Tmin is used for sorting therefore we need to set it far away
        inter_tmin[i]           = 123456789;
        inter_num_samples[i]    = 0;
        inter_id[i]             = i;
    }

    __syncthreads();

    // Each block processes one ray
    Vec origin;
    Vec direction;
    for(int d = 0; d < D; ++d)
    {
        origin(d)   = ray_origin(ray_id, d);
        direction(d)= ray_direction(ray_id, d);
    }
    
    int node_id = 0;
    Vec box_min, box_max;
    for(int d = 0; d < D; ++d)
    {
        box_min(d) = tree_roi_min(d);
        box_max(d) = tree_roi_max(d);
    }

    for(int i = threadIdx.x; i < 1; i+= blockDim.x)
    {
        auto [hit, tmin, tmax] = 
        IntersectBoxRayPrecise(box_min.data(), box_max.data(), origin.data(), direction.data(), D);

        if(tmax - tmin < 1e-5)
        {
            continue;
        }
        if(hit)
        {
            auto index = atomicAdd(&num_intersections, 1);
            CUDA_KERNEL_ASSERT(index < num_intersections);
            inter_tmin[index] = tmin;
            inter_tmax[index] = tmax;
            inter_node_id[index] = node_id;

            float diag                  = (box_max - box_min).norm();
            float dis                   = tmax - tmin;
            float rel_dis               = dis/diag;
            int num_samples             = iCeil(rel_dis * max_samples_per_node);
            inter_num_samples[index]    = num_samples;

            atomicAdd(&num_samples_of_ray, num_samples);
        }

    }

    __syncthreads();

    if(threadIdx.x == 0)
    {

        global_sample_offset = atomicAdd(out_num_samples, num_samples_of_ray);
        atomicMax(out_num_samples + 1, num_samples_of_ray);
        CUDA_KERNEL_ASSERT(global_sample_offset + num_samples_of_ray < out_weight.sizes[0]);        
    }
    __syncthreads();

    // caffe2::bitonicSort<CompareInterval, float, int, max_intersections, ThreadsPerBlock>(inter_tmin, inter_id, CompareInterval());

    // TODO: more efficient scan
    for (int i = threadIdx.x; i < max_intersections; i += blockDim.x)
    {
        int count = 0;
        for (int j = 0; j < i; ++j)
        {
            int sorted_id   = inter_id[j];
            int num_samples = inter_num_samples[sorted_id];
            count += num_samples;
        }
        inter_num_samples_scan[i] = count;
    }

    __syncthreads();


    for(int iid = warp_id; iid < num_intersections; iid+= num_warps)
    {
        int sorted_id           = inter_id[iid];
        float tmin              = inter_tmin[iid];
        int num_samples_scan    = inter_num_samples_scan[iid];
        int node_id             = inter_node_id[sorted_id];
        float tmax              = inter_tmax[sorted_id];
        int num_samples         = inter_num_samples[sorted_id];

        CUDA_KERNEL_ASSERT(num_samples <= max_samples_per_node);
        if (num_samples == 0) continue;

        float dis       = tmax - tmin;
        float weight    = dis/num_samples;
        float step      = (tmax - tmin)/num_samples;

        int out_sample_index = 0;
        out_sample_index = global_sample_offset + num_samples_scan;

        for(int j = lane_id; j < num_samples; j+= 32)
        {
            int global_sample_idx = out_sample_index + j;
            float t1              = tmin + j * step;
            float t2              = tmin + (j + 1) * step;

            float a = 0.5;
            if (sample_rnd)
            {
                a = sample_rnd[global_sample_idx];
            }
            float t        = t1 * (1 - a) + t2 * a;
            Vec global_pos = origin + t * direction;
            for (int d = 0; d < D; ++d)
            {
                out_global_coordinates(global_sample_idx, d) = global_pos(d);
            }
            out_ray_t(global_sample_idx)        = t;
            out_weight(global_sample_idx)       = weight;
            out_node_id(global_sample_idx)      = node_id;
            out_ray_index(global_sample_idx)    = ray_id;
            out_ray_local_id(global_sample_idx) = num_samples_scan + j;
        }

        __syncwarp();
        for (int j = lane_id; j < num_samples; j += 32)
        {
            float t = out_ray_t(out_sample_index + j);

            float t_half_min;
            if (j == 0)
                t_half_min = tmin;
            else
                t_half_min = (out_ray_t(out_sample_index + j - 1) + t) * 0.5f;

            float t_half_max;
            if (j == num_samples - 1)
                t_half_max = tmax;
            else
                t_half_max = (out_ray_t(out_sample_index + j + 1) + t) * 0.5f;

            out_weight(out_sample_index + j) = t_half_max - t_half_min;
        }
    }
    
}

SampleList HyperTreeBaseImpl::CreateSamplesForRays_area(const RayList& rays, int max_samples_per_node, bool interval_jitter, vec3 tree_roi_min, vec3 tree_roi_max)
{
    CHECK(rays.direction.is_cuda());
    CHECK(node_position_min.is_cuda());

    int predicted_samples = iCeil(rays.size() * max_samples_per_node * pow(1, 1.0/D()));

    SampleList list;
    list.Allocate(predicted_samples, D(), node_position_min.device());


    torch::Tensor interval_rnd = interval_jitter ? torch::rand_like(list.weight) : torch::Tensor();
    float* interval_rnd_ptr    = interval_jitter ? interval_rnd.data_ptr<float>() : nullptr;

    auto out_num_samples_max_per_ray = torch::zeros({2}, node_position_min.options().dtype(torch::kInt32));


    switch(D())
    {
        case 3:
        // TO DO
        // To test without :: whether is global or local

        ::ComputeRaySamples_area<3, 128><<<rays.size(), 128>>>(
                this, rays.origin, rays.direction, interval_rnd_ptr,
                out_num_samples_max_per_ray.data_ptr<int>(), list.global_coordinate, list.weight, list.ray_t,
                list.ray_index, list.local_index_in_ray, list.node_id, max_samples_per_node, tree_roi_min, tree_roi_max);
            break;
        default:
            CHECK(false);
    }

    out_num_samples_max_per_ray = out_num_samples_max_per_ray.cpu();
    int actual_samples          = out_num_samples_max_per_ray.data_ptr<int>()[0];
    list.max_samples_per_ray    = out_num_samples_max_per_ray.data_ptr<int>()[1];
    list.Shrink(actual_samples);
    CHECK_LE(actual_samples, predicted_samples);

    {
        // Use this method for position computation to get the correct positional gradient
        auto origin2           = torch::index_select(rays.origin, 0, list.ray_index);
        auto dir2              = torch::index_select(rays.direction, 0, list.ray_index);
        auto pos2              = origin2 + dir2 * list.ray_t.unsqueeze(1);
        list.global_coordinate = pos2;
    }
    


    CUDA_SYNC_CHECK_ERROR();
    return list;
}



torch::Tensor HyperTreeBaseImpl::CreateSamplesRandomly(torch::Tensor node_id,const int num_of_samples_per_edge)
{
    // CHECK(num_of_samples_per_edge > 0);
    CHECK(node_position_min.is_cuda());
    int predicted_samples = pow(num_of_samples_per_edge,D()) * NumActiveNodes();
    node_id = node_id.to(torch::kInt);

    // auto device = node_position_min.device();
    // SampleList list;
    int N = NumActiveNodes();
    // list.Allocate(predicted_samples, D(), device);
    torch::Tensor rand_value = torch::rand({N, num_of_samples_per_edge, num_of_samples_per_edge, num_of_samples_per_edge,D()},torch::TensorOptions(node_position_min.device()).dtype(torch::kFloat));
    #if 0
    {
        Printf("Should not run\n");
        auto result_position = torch::zeros({N, num_of_samples_per_edge, num_of_samples_per_edge, num_of_samples_per_edge,D()},torch::TensorOptions(node_position_min.device()).dtype(torch::kFloat));
        ::RandomGlobalSamples<3><<<N, 128>>> (this, node_id, ivec3(num_of_samples_per_edge,num_of_samples_per_edge,num_of_samples_per_edge),rand_value,result_position );
    }
    #endif

    torch::Tensor active_node_min = torch::index_select(node_position_min, 0, active_node_ids);

    torch::Tensor sizes = torch::index_select(node_position_max, 0, active_node_ids) - active_node_min;

    auto result_position2 = active_node_min.unsqueeze(1).unsqueeze(1).unsqueeze(1) + rand_value * sizes.unsqueeze(1).unsqueeze(1).unsqueeze(1);
    // auto result_position2 = active_node_min + rand_value * sizes;
    // printf("result postion is ");

    // auto res = result_position - result_position2;
    // PrintTensorInfo(res);
    
    CUDA_SYNC_CHECK_ERROR();
    return result_position2;
}

static __global__ void CountSamplesPerNode(StaticDeviceTensor<long, 1> node_id,
                                           StaticDeviceTensor<int, 1> out_node_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= node_id.sizes[0]) return;
    int target_idx = node_id(tid);
    atomicAdd(&out_node_count(target_idx), 1);
}

// static __global__ void ComputeBoundWeight(StaticDeviceTensor<long,1> node_id, 
//                                             StaticDeviceTensor<long, 1> in_roi_node,
//                                             StaticDeviceTensor<long, 1> weight_bound_index)
// {
//     // int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     int group_id = blockIdx.x;
//     int local_tid = threadIdx.x;
//     // if(tid >= node_id.sizes[0]) return;
//     int group_size = node_id.sizes[0];
//     // for(int i = 0; i < in_roi_node.sizes[0]; ++i)
//     for(int i = local_tid; i < group_size; i+=blockDim.x)
//     {
//         if(! (node_id(i) == in_roi_node(group_id)))
//         {
//             weight_bound_index(i) = i;
//         }
//     }

//     // int group_id = blockIdx.x;
//     // int local_tid = threadIdx.x;
//     // int group_size = in_roi_node.sizes[0];
//     // for(int i = local_tid; i < group_size; i+= blockDim.x)
//     // {
//     //     if(!((node_id(group_id) == in_roi_node(i))))
//     //     {
//     //         weight_bound_index(group_id) = group_id;
//     //     }
//     // }

// }

static __global__ void ComputeBoundWeight(StaticDeviceTensor<long,1> node_id, 
                                            StaticDeviceTensor<long, 1> in_roi_node,
                                            StaticDeviceTensor<long, 1> weight_bound_index)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid >= node_id.sizes[0]) return;
    for(int i = 0; i < in_roi_node.sizes[0]; ++i)
    {
        if((node_id(tid) == in_roi_node(i)))
        {
            weight_bound_index(tid) = tid;
        }
    }
}

torch::Tensor HyperTreeBaseImpl::ComputeBoundWeight(torch::Tensor node_id, torch::Tensor in_roi_node, bool out_selected)
{
    CHECK(node_id.is_cuda());
    long node_len = node_id.sizes()[0];
    // auto device = node_id.device();


    torch::Tensor weight_bound_index = -torch::ones({node_len},torch::TensorOptions(node_id.device()).dtype(torch::kLong));
    ::ComputeBoundWeight<<<iDivUp(node_len, 256), 256>>>(node_id, in_roi_node, weight_bound_index);
    if(out_selected)
    {
        weight_bound_index = torch::masked_select(weight_bound_index, weight_bound_index >= 0);
    }
    return weight_bound_index;

}

static __global__ void ComputeRatio(StaticDeviceTensor<float, 2> weight_in, StaticDeviceTensor<float,2 > weight_all,
                                    StaticDeviceTensor<float, 2> ratio, StaticDeviceTensor<float, 2> ratio_inv)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= weight_in.sizes[1]) return;
    if(weight_all(0, tid) > 0)
    {
        ratio(0, tid) = weight_in(0, tid)/ weight_all(0, tid);
    }
    if(weight_in(0, tid) > 0)
    {
        ratio_inv(0, tid) = weight_all(0, tid)/ weight_in(0,tid);
    }
}

std::tuple<torch::Tensor, torch::Tensor> HyperTreeBaseImpl::ComputeRatio(torch::Tensor weight_in, torch::Tensor weight_all)
{
    CHECK(weight_in.is_cuda());
    CHECK(weight_in.dim() == 2);

    // auto device = weight_in.device();
    long weight_len = weight_in.sizes()[1];

    torch::Tensor ratio = torch::ones_like(weight_in);
    torch::Tensor ratio_inv = torch::zeros_like(weight_in);
    ::ComputeRatio<<<iDivUp(weight_len, 256), 256>>>(weight_in, weight_all, ratio, ratio_inv);
    return {ratio, ratio_inv};
}
static __global__ void ComputedPaddedCount(StaticDeviceTensor<int, 1> node_count,
                                           StaticDeviceTensor<int, 1> out_node_count_padded, int group_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= node_count.sizes[0]) return;
    int count                  = node_count(tid);
    int padded_count           = iAlignUp(count, group_size);
    out_node_count_padded(tid) = padded_count;
}

static __global__ void ComputeIndexOrder(StaticDeviceTensor<long, 1> node_id,
                                         StaticDeviceTensor<long, 1> samples_per_node_scan,
                                         StaticDeviceTensor<int, 1> current_node_elements,
                                         StaticDeviceTensor<long, 1> per_group_node_id,
                                         StaticDeviceTensor<int, 1> src_indices,
                                         StaticDeviceTensor<float, 1> padding_weights, int group_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= node_id.sizes[0]) return;
    int nid    = node_id(tid);
    int start  = nid == 0 ? 0 : samples_per_node_scan(nid - 1);
    int offset = atomicAdd(&current_node_elements(nid), 1);

    if (offset % group_size == 0)
    {
        // The first sample of each group writes the node id!
        int k = offset / group_size;

        per_group_node_id(start / group_size + k) = nid;
    }
    src_indices(start + offset)     = tid;
    padding_weights(start + offset) = 1;
}

NodeBatchedSamples HyperTreeBaseImpl::GroupSamplesPerNodeGPU(const SampleList& samples, int group_size)
{
    // printf("test this function\n");
    // auto device = samples.global_coordinate.device();
    CHECK_EQ(samples.global_coordinate.device(), active_node_ids.device());
    int num_samples = samples.size();
    int num_nodes   = NumNodes();
    // CHECK_GT(num_samples, 0);

    // torch::Tensor num_samples_per_node = torch::zeros({num_nodes}, torch::TensorOptions(torch::kInt)).to(at::kCUDA);
    torch::Tensor num_samples_per_node = torch::zeros({num_nodes}, torch::TensorOptions(samples.global_coordinate.device()).dtype(torch::kInt));

    if (num_samples > 0)
    {
        CountSamplesPerNode<<<iDivUp(num_samples, 256), 256>>>(samples.node_id, num_samples_per_node);
    }


    torch::Tensor num_samples_per_node_padded = torch::zeros_like(num_samples_per_node);

    ComputedPaddedCount<<<iDivUp(num_nodes, 256), 256>>>(num_samples_per_node, num_samples_per_node_padded, group_size);

    auto samples_per_node_scan = torch::cumsum(num_samples_per_node_padded, 0);

    int num_output_samples     = samples_per_node_scan.slice(0, num_nodes - 1, num_nodes).item().toLong();


    SAIGA_ASSERT(num_output_samples % group_size == 0);

    int num_groups = num_output_samples / group_size;

    torch::Tensor current_node_elements = torch::zeros({num_nodes}, torch::TensorOptions(samples.global_coordinate.device()).dtype(torch::kInt));
    torch::Tensor per_group_node_id     = torch::zeros({num_groups}, torch::TensorOptions(samples.global_coordinate.device()).dtype(torch::kLong));
    torch::Tensor src_indices = torch::zeros({num_output_samples}, torch::TensorOptions(samples.global_coordinate.device()).dtype(torch::kInt));
    torch::Tensor padding_weights =
        torch::zeros({num_output_samples}, torch::TensorOptions(samples.global_coordinate.device()).dtype(torch::kFloat));
    // printf("test group samples per node 3\n");

    if (num_samples > 0)
    {
        ComputeIndexOrder<<<iDivUp(num_samples, 256), 256>>>(samples.node_id, samples_per_node_scan,
                                                             current_node_elements, per_group_node_id, src_indices,
                                                             padding_weights, group_size);
    }
    // printf("test group samples per node 4\n");


    NodeBatchedSamples result;
    result.global_coordinate = torch::index_select(samples.global_coordinate, 0, src_indices);
    result.global_coordinate = result.global_coordinate.reshape({-1, group_size, D()});

    result.mask = padding_weights.reshape({-1, group_size, 1});

    result.integration_weight = torch::index_select(samples.weight, 0, src_indices) * padding_weights;
    result.integration_weight = result.integration_weight.reshape({-1, group_size, 1});

    result.node_ids = per_group_node_id;

    result.ray_index = torch::index_select(samples.ray_index, 0, src_indices);
    result.ray_index = result.ray_index.reshape({-1, group_size, 1});


    // auto test_coord = torch::index_select(samples.global_coordinate, 0, src_indices);

    // auto test_nodeid = torch::index_select(samples.node_id, 0, src_indices);
    // test_nodeid = test_nodeid.reshape({-1, group_size, 1});
    // test_nodeid = test_nodeid * result.mask;
    // for(int index = 0; index < test_nodeid.size(0); ++index)
    // {
    //     auto slice = test_nodeid.slice(0, index, index+1);
    //     auto [test,test1,test2] = torch::unique_consecutive(slice);
    //     if(test.size(0) > 1)
    //     {
    //         if(test.size(0) > 2)
    //         {
    //             printf("got an error\n");
    //         }
    //         else 
    //         {
    //             if( ( test[0].item().toFloat() == 0) || (test[1].item().toFloat() == 0 ))
    //             {
    //                 continue;
    //             }
    //             else
    //             {
    //                 PrintTensorInfo(test);
    //             }
    //         }
    //     }
    // }

    if (samples.local_index_in_ray.defined())
    {
        result.sample_index_in_ray = torch::index_select(samples.local_index_in_ray, 0, src_indices);
        result.sample_index_in_ray = result.sample_index_in_ray.reshape({-1, group_size, 1});
    }
    CUDA_SYNC_CHECK_ERROR();
    return result;
}


template <int D>
static __global__ void VolumeSamples(Eigen::Vector<int, D> size, StaticDeviceTensor<float, 2> out_position,
                                     StaticDeviceTensor<long, 1> out_index, bool swap_xy)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_position.sizes[0]) return;

    Eigen::Vector<int, D> c;
    int tmp = tid;

    if (swap_xy)
    {
        for (int d = D - 1; d >= 0; --d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }
    else
    {
        for (int d = 0; d < D; ++d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }

    Eigen::Vector<float, D> ones = Eigen::Vector<float, D>::Ones();
    Eigen::Vector<float, D> pos =
        (c.template cast<float>().array() / (size.template cast<float>() - ones).array()).eval();
    Eigen::Vector<float, D> global_coordinates = ((pos * 2).eval() - ones).eval();

    for (int d = 0; d < D; ++d)
    {
        out_position(tid, d) = global_coordinates(d);
    }
    out_index(tid) = tid;
}

SampleList HyperTreeBaseImpl::UniformPhantomSamplesGPU(Eigen::Vector<int, -1> size, bool swap_xy)
{
    int num_samples = size.array().prod();
    SampleList result;
    result.Allocate(num_samples, D(), device());

    printf("sample volume \n");
    PrintTensorInfo(result.global_coordinate);


    VolumeSamples<3><<<iDivUp(num_samples, 256), 256>>>(size, result.global_coordinate, result.ray_index, swap_xy);

    auto test1 = result.global_coordinate.slice(1,0,1);
    PrintTensorInfo(test1);
    test1 = result.global_coordinate.slice(1,1,2);
    PrintTensorInfo(test1);

    test1 = result.global_coordinate.slice(1,2,3);
    PrintTensorInfo(test1);

    std::tie(result.node_id, result.weight) = NodeIdForPositionGPU(result.global_coordinate);
    CUDA_SYNC_CHECK_ERROR();
    return result;
}

template <int D> 
static __global__ void VolumeSamplesbySlice(Eigen::Vector<int ,D> size, StaticDeviceTensor<float, 2> out_position, StaticDeviceTensor<long, 1> out_index, 
                                            bool swap_xy, Eigen::Vector<float, D> roi_min, Eigen::Vector<float, D> step_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= out_position.sizes[0]) return;

    Eigen::Vector<int, D> c;
    int tmp = tid;
    if (swap_xy)
    {
        for(int d = D-1; d>=0; --d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }
    else
    {
        for(int d = 0; d< D; ++d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }

    // Eigen::Vector<float, D> global_coordinates = roi_min.template cast<float>().array + 
    //                     step_size.template cast<float>().array * c.template cast<float>().array;

    // Eigen::Vector<float, D> global_coordinates = (roi_min +  step_size  * c.template cast<float>().array).eval();

    // Eigen::Vector<float, D> pos =
    //     (c.template cast<float>().array() / (size.template cast<float>() - ones).array()).eval();

    // Eigen::Vector<float, D> ones = Eigen::Vector<float, D>::Ones();
    // Eigen::Vector<float, D> pos =
    //     (c.template cast<float>().array() / (size.template cast<float>() - ones).array()).eval();
    // Eigen::Vector<float, D> global_coordinates = ((pos * 2).eval() - ones).eval();

    Eigen::Vector<float, D> global_coordinates = (roi_min.template cast<float>().array() +  step_size.template cast<float>().array() *c.template cast<float>().array() ).eval();

    // Eigen::Vector<float, D> global_coordinates = roi_min.template cast<float>().array() +  step_size.template cast<float>().array() *c.template cast<float>().array();


    for(int d = 0; d < D; ++d)
    {
        out_position(tid, d) = global_coordinates(d);
    }
    out_index(tid) = tid;

}

SampleList HyperTreeBaseImpl::UniformPhantomSamplesGPUbySlice(Eigen::Vector<int, -1> size, bool swap_xy, Eigen::Vector<float, 3> roi_min, Eigen::Vector<float, 3> step_size)
{
    int num_samples = size.array().prod();
    SampleList result;
    result.Allocate(num_samples, D(), device());


    VolumeSamplesbySlice<3><<<iDivUp(num_samples, 256), 256>>>(size, result.global_coordinate, result.ray_index, swap_xy, roi_min, step_size);

    std::tie(result.node_id, result.weight) = NodeIdForPositionGPU(result.global_coordinate);

    CUDA_SYNC_CHECK_ERROR();
    return result;
}


template <int D> 
static __global__ void VolumeSamplesbySlice_global(Eigen::Vector<int ,D> size, StaticDeviceTensor<float, 2> out_position,
                                            bool swap_xy, Eigen::Vector<float, D> roi_min, Eigen::Vector<float, D> step_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= out_position.sizes[0]) return;

    Eigen::Vector<int, D> c;
    int tmp = tid;
    if (swap_xy)
    {
        for(int d = D-1; d>=0; --d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }
    else
    {
        for(int d = 0; d< D; ++d)
        {
            c(d) = tmp % size(d);
            tmp /= size(d);
        }
    }


    Eigen::Vector<float, D> global_coordinates = (roi_min.template cast<float>().array() +  step_size.template cast<float>().array() *c.template cast<float>().array() ).eval();

    // Eigen::Vector<float, D> global_coordinates = roi_min.template cast<float>().array() +  step_size.template cast<float>().array() *c.template cast<float>().array();


    for(int d = 0; d < D; ++d)
    {
        out_position(tid, d) = global_coordinates(d);
    }
}


torch::Tensor HyperTreeBaseImpl::UniformPhantomSamplesGPUSlice_global(Eigen::Vector<int, -1> size, bool swap_xy, Eigen::Vector<float, 3> roi_min, Eigen::Vector<float, 3> step_size)
{
    int num_samples = size.array().prod();
    torch::Tensor global_coordinate  = torch::empty({num_samples, D()}, torch::TensorOptions(torch::kFloat32)).to(device());

    VolumeSamplesbySlice_global<3><<<iDivUp(num_samples, 256), 256>>>(size, global_coordinate,  swap_xy, roi_min, step_size);
    CUDA_SYNC_CHECK_ERROR();

    return global_coordinate;
}
template <int D>
__device__ inline int ActiveNodeIdForGlobalPosition(DeviceHyperTree<D> tree, float* global_position, bool use_quad)
{
    // constexpr int NS = 1 << D;
    int NS;
    if(use_quad)
    {
        NS              = 1 << (D-1);
    }
    else
    {
        NS              = 1 << D;
    }
    int current_node = 0;
    while (true)
    {
        int node_id = current_node;
        if (tree.node_active(current_node) > 0)
        {
            break;
        }
        // Not active -> decent into children which contains the sample
        for (int cid = 0; cid < NS; ++cid)
        {
            int c = tree.node_children(current_node, cid);
            if (c == -1)
            {
                // This sample is not inside an active node
                return -1;
            }
            CUDA_KERNEL_ASSERT(c >= 0);
            CUDA_KERNEL_ASSERT(c != node_id);
            // float* pos_min = &tree.node_position_min(c, 0);
            // float* pos_max = &tree.node_position_max(c, 0);
            // if (BoxContainsPoint(pos_min, pos_max, global_position, D))
            // {
            //     // printf("pos %d node active %d %f %f %f %f %f %f %f %f %f\n", c,tree.node_active(current_node), global_position[0],
            //     //         global_position[1], global_position[2], pos_min[0], pos_min[1], pos_min[2], pos_max[0], pos_max[1], pos_max[2]);
            //     current_node = c;
            //     break;
            // }
            // delete pos_min;
            // delete pos_max;

            if( tree.node_position_min(c,0) <= global_position[0] && tree.node_position_max(c,0) >= global_position[0] &&
                tree.node_position_min(c,1) <= global_position[1] && tree.node_position_max(c,1) >= global_position[1] &&
                tree.node_position_min(c,2) <= global_position[2] && tree.node_position_max(c,2) >= global_position[2])
            {
                current_node = c;
                break;
            }

        }
        // printf("current node %d %d\n", current_node, node_id);
        if (current_node == node_id)
        {
            return -1;
        }
        CUDA_KERNEL_ASSERT(current_node != node_id);
    }
    return current_node;
}

template <int D>
inline int ActiveNodeIdForGlobalPosition_CPU(HyperTreeBase tree, float * global_position, bool use_quad)
{
   // constexpr int NS = 1 << D;
   int NS;
   if(use_quad)
   {
       NS              = 1 << (D-1);
   }
   else
   {
       NS              = 1 << D;
   }
   int current_node = 0;
//    std::cout << "node NS " << NS << std::endl;
//    printf("global position is %f %f %f\n", global_position[0], global_position[1], global_position[2]);
   while (true)
   {
        int node_id = current_node;
        if (tree->node_active.data_ptr<int>()[current_node * tree->node_active.stride(0)] > 0)
        {
           break;
            // float* pos_min = tree->node_position_min.data_ptr<float>();
            // float* pos_max = tree->node_position_max.data_ptr<float>();

        }
        // else
        // {
            // Not active -> decent into children which contains the sample
            for (int cid = 0; cid < NS; ++cid)
            {
                //    int c = tree.node_children(current_node, cid);
                int c =tree->node_children.data_ptr<int>()[current_node* tree->node_children.stride(0) + cid * tree->node_children.stride(1) ];
                if (c == -1)
                {
                    // This sample is not inside an active node
                    return -1;
                }
                CUDA_KERNEL_ASSERT(c >= 0);
                CUDA_KERNEL_ASSERT(c != node_id);
                //    float* pos_min = &tree.node_position_min(c, 0);
                //    float* pos_max = &tree.node_position_max(c, 0);
                float* pos_min = tree->node_position_min.data_ptr<float>() + c * tree->node_position_min.stride(0);
                float* pos_max = tree->node_position_max.data_ptr<float>() + c * tree->node_position_max.stride(0);
                // printf("current node children %d pos %f %f %f %f %f %f", c, pos_min[0], pos_min[1], pos_min[2], pos_max[0], pos_max[1], pos_max[2]);

                if (BoxContainsPoint(pos_min, pos_max, global_position, D))
                {
                    current_node = c;
                    printf("current node children %d pos %f %f %f %f %f %f", c, pos_min[0], pos_min[1], pos_min[2], pos_max[0], pos_max[1], pos_max[2]);
                    break;
                }
            }
        // }


       if (current_node == node_id)
       {
           return -1;
       }
       CUDA_KERNEL_ASSERT(current_node != node_id);
   }
//    printf("return with current node %d \n", current_node);
   return current_node;
}

template <int D>
static __global__ void ComputeNodeId(StaticDeviceTensor<float, 2> global_samples, DeviceHyperTree<D> tree,
                                     StaticDeviceTensor<long, 1> out_index, StaticDeviceTensor<float, 1> out_mask, bool use_quad)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= global_samples.sizes[0]) return;
    float* position  = &global_samples(tid, 0);
    int current_node = ActiveNodeIdForGlobalPosition(tree, position, use_quad);

    if (current_node == -1)
    {
        // just use the first active node id
        out_index(tid) = tree.active_node_ids(0);
        out_mask(tid)  = 0;
    }
    else
    {
        out_index(tid) = current_node;
        out_mask(tid)  = 1;
    }
}

template <int D> 
void ComputeNodeId_CPU(torch::Tensor global_samples, HyperTreeBase tree,
    torch::Tensor out_index, torch::Tensor out_mask, bool use_quad)
{
    std::cout << "stride is " << global_samples.stride(0) << " " << global_samples.stride(1) << std::endl;
    for(int i = 0; i < global_samples.size(0);++i)
    {
        float* position = global_samples.data_ptr<float>() ;
        position += i *global_samples.stride(0);
        int current_node = ActiveNodeIdForGlobalPosition_CPU<3>(tree, position, true);
        // char c = getchar();
    }
}

std::tuple<torch::Tensor, torch::Tensor> HyperTreeBaseImpl::NodeIdForPositionGPU(torch::Tensor global_samples)
{
    CHECK(global_samples.is_cuda());
    CHECK(node_position_min.is_cuda());
    auto samples_linear = global_samples.reshape({-1, D()});
    int num_samples     = samples_linear.size(0);

    auto result_node_id = torch::zeros({num_samples}, global_samples.options().dtype(torch::kLong));
    auto result_mask    = torch::zeros({num_samples}, global_samples.options().dtype(torch::kFloat));

    CHECK(samples_linear.is_contiguous());

    CHECK_EQ(D(), 3);
    if (num_samples > 0)
    {
        ComputeNodeId<3><<<iDivUp(num_samples, 256), 256>>>(samples_linear, this, result_node_id, result_mask, use_quad_tree);
        // ComputeNodeId_CPU<3>(samples_linear, old_tree, result_node_id, result_mask, use_quad_tree);

    }
    CUDA_SYNC_CHECK_ERROR();
    return {result_node_id, result_mask};
}

template <int D>
static __global__ void InterpolateGridForInactiveNodes(DeviceHyperTree<D> tree,
                                                       StaticDeviceTensor<float, 5> active_grid,
                                                       StaticDeviceTensor<float, 5> out_full_grid, bool use_quad)
{
    int node_id   = blockIdx.x;
    int thread_id = threadIdx.x;
    CUDA_KERNEL_ASSERT(node_id < active_grid.sizes[0]);

    using Vec = vec3;
    Vec box_min;
    Vec box_max;
    for (int d = 0; d < D; ++d)
    {
        box_min(d) = tree.node_position_min(node_id, d);
        box_max(d) = tree.node_position_max(node_id, d);
    }
    Vec box_size = box_max - box_min;

    if (!tree.node_active(node_id))
    {
        //        printf("processing active node %d\n", node_id);
        ivec3 grid_size(11, 11, 11);
        int total_cells = grid_size.array().prod();
        for (int cell_id = thread_id; cell_id < total_cells; cell_id += blockDim.x)
        {
            ivec3 res;
            int tmp = cell_id;
            for (int d = 0; d < D; ++d)
            {
                res(d) = tmp % grid_size(d);
                tmp /= grid_size(d);
            }

            // range [0,1]
            Vec local_coords = res.array().cast<float>() / (grid_size - ivec3::Ones()).array().cast<float>();


            // convert into global
            Vec c             = local_coords;
            c                 = c.array() * box_size.array();
            Vec global_coords = (c + box_min);

            int target_node = ActiveNodeIdForGlobalPosition(tree, global_coords.data(), use_quad);


            // convert into local space of target node
            Vec target_box_min;
            Vec target_box_max;
            for (int d = 0; d < D; ++d)
            {
                target_box_min(d) = tree.node_position_min(target_node, d);
                target_box_max(d) = tree.node_position_max(target_node, d);
            }
            Vec target_box_size = target_box_max - target_box_min;
            c                   = global_coords;
            c                   = c - target_box_min;
            c                   = c.array() / target_box_size.array();
            // range [0,1]
            Vec target_local_coords_01 = c;

            // compute nearest neighbor of target
            Vec feature_coord    = target_local_coords_01.array() * (grid_size - ivec3::Ones()).array().cast<float>();
            ivec3 feature_coordi = feature_coord.array().round().cast<int>();


            // copy into target
            for (int c = 0; c < out_full_grid.sizes[1]; ++c)
            {
                out_full_grid(node_id, c, res(2), res(1), res(0)) =
                    active_grid(target_node, c, feature_coordi(2), feature_coordi(1), feature_coordi(0));
            }
        }
    }
}

torch::Tensor HyperTreeBaseImpl::InterpolateGridForInactiveNodes(torch::Tensor active_grid)
{
    std::cout << "HyperTreeBaseImpl::InterpolateGridForInactiveNodes" << std::endl;

    // float [num_nodes, 8, 11, 11, 11]
    torch::Tensor interpolated = active_grid.clone();
    CHECK_EQ(D(), 3);
    CHECK_EQ(interpolated.size(2), 11);

    PrintTensorInfo(active_grid);
    ::InterpolateGridForInactiveNodes<3><<<NumNodes(), 128>>>(this, active_grid, interpolated, use_quad_tree);
    CUDA_SYNC_CHECK_ERROR();
    return interpolated;
}

template <int D>
static __global__ void UniformGlobalSamples(DeviceHyperTree<D> tree, StaticDeviceTensor<int, 1> node_ids,
                                            ivec3 grid_size, StaticDeviceTensor<float, 5> out_position)
{
    CUDA_KERNEL_ASSERT(blockIdx.x < node_ids.sizes[0]);

    int node_id   = node_ids((int)blockIdx.x);
    int thread_id = threadIdx.x;

    vec3 pos_min = tree.PositionMin(node_id);
    vec3 pos_max = tree.PositionMax(node_id);
    vec3 size    = pos_max - pos_min;

    int total_cells = grid_size.array().prod();
    for (int cell_id = thread_id; cell_id < total_cells; cell_id += blockDim.x)
    {
        ivec3 res;
        int tmp = cell_id;
        for (int d = 0; d < D; ++d)
        {
            res(d) = tmp % grid_size(d);
            tmp /= grid_size(d);
        }

        // in [0,1]
        vec3 local_position = res.cast<float>().array() / (grid_size - ivec3::Ones()).cast<float>().array();

        vec3 global_position = local_position.array() * size.array();
        global_position += pos_min;

        for (int d = 0; d < D; ++d)
        {
            out_position((int)blockIdx.x, res(2), res(1), res(0), d) = global_position(d);
        }
    }
}
torch::Tensor HyperTreeBaseImpl::UniformGlobalSamples(torch::Tensor node_id, int grid_size)
{
    CHECK_EQ(node_id.dim(), 1);

    int N   = node_id.size(0);
    node_id = node_id.to(torch::kInt);

    auto result_position = torch::zeros({N, grid_size, grid_size, grid_size, D()}, node_position_min.options());
    ::UniformGlobalSamples<3><<<N, 128>>>(this, node_id, ivec3(grid_size, grid_size, grid_size), result_position);

    return result_position;
}

template <int D> __device__ inline
bool in_roi(DeviceHyperTree<D> tree, int node_id, vec3 roi_min, vec3 roi_max)
{
    vec3 node_min = tree.PositionMin(node_id);
    vec3 node_max = tree.PositionMax(node_id);
    vec3 node_mid = node_min + node_max;
    if( ((node_min(0) > roi_min(0) &&  node_min(0) < roi_max(0)) || 
        (node_max(0) > roi_min(0) && node_max(0) < roi_max(0)) ||
        (node_mid(0) > roi_min(0) && node_mid(0) < roi_max(0))) &&
        ((node_min(1) > roi_min(1) &&  node_min(1) < roi_max(1)) || 
        (node_max(1) > roi_min(1) && node_max(1) < roi_max(1)) ||
        (node_mid(1) > roi_min(1) && node_mid(1) < roi_max(1))) &&
        ((node_min(2) > roi_min(2) &&  node_min(2) < roi_max(2)) || 
        (node_max(2) > roi_min(2) && node_max(2) < roi_max(2)) ||
        (node_mid(2) > roi_min(2) && node_mid(2) < roi_max(2))) )
        {
            return true;
        }
        else{
            return false;
        }
}

template <int D> __device__ inline
bool in_roi_quad(DeviceHyperTree<D> tree, int node_id, vec3 roi_min, vec3 roi_max, float epsilon)
{
    float roi_min_y = roi_min(1) - epsilon;
    float roi_max_y = roi_max(1) + epsilon;
    vec3 node_min = tree.PositionMin(node_id);
    vec3 node_max = tree.PositionMax(node_id);
    vec3 node_mid = node_min + node_max;
    // float roi_min_x = roi_min(0) ;
    // float roi_max_x = roi_max(0) ;
    // float roi_min_z = roi_min(2) ;
    // float roi_max_z = roi_max(2) ;
    if( ((node_min(0) > roi_min(0) &&  node_min(0) < roi_max(0)) || 
        (node_max(0) > roi_min(0) && node_max(0) < roi_max(0)) ||
        (node_mid(0) > roi_min(0) && node_mid(0) < roi_max(0))) &&
        ((node_min(1) > roi_min_y &&  node_min(1) < roi_max_y) || 
        (node_max(1) > roi_min_y && node_max(1) < roi_max_y) ||
        (node_mid(1) > roi_min_y && node_mid(1) < roi_max_y)) &&
        ((node_min(2) > roi_min(2) &&  node_min(2) < roi_max(2)) || 
        (node_max(2) > roi_min(2) && node_max(2) < roi_max(2)) ||
        (node_mid(2) > roi_min(2) && node_mid(2) < roi_max(2))) )
        {
            return true;
        }
        else{
            return false;
        }

}

template <int D>
static __global__ void NodeNeighborSamples(DeviceHyperTree<D> tree, Eigen::Vector<int, D> size, float epsilon,
                                           int* out_num_samples, StaticDeviceTensor<float, 2> out_global_coordinates,
                                           StaticDeviceTensor<float, 1> out_weight,
                                           StaticDeviceTensor<float, 1> out_ray_t,
                                           StaticDeviceTensor<long, 1> out_ray_index,
                                           StaticDeviceTensor<long, 1> out_ray_local_id,
                                           StaticDeviceTensor<long, 1> out_node_id,
                                            vec3 roi_min, vec3 roi_max, bool use_roi, bool use_quad)
{
    int node_id  = tree.active_node_ids((int)blockIdx.x);
    int local_id = threadIdx.x;


    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid >= out_position.sizes[0]) return;

    vec3 node_min = tree.PositionMin(node_id);
    vec3 node_max = tree.PositionMax(node_id);
    vec3 box_size = node_max - node_min;

    for (int side = 0; side < D; ++side)
    {
        int side_a = (side + 1) % 3;
        int side_b = (side + 2) % 3;

        for (int front = 0; front < 2; ++front)
        {
            vec3 add_eps(0, 0, 0);

            add_eps(side) = epsilon * (front ? 1 : -1);
            float rem     = (front ? 1 : 0);

            for (int sid = local_id; sid < size(side_a) * size(side_b); sid += blockDim.x)
            {
                // float x = 0;
                // float y = (sid % size(1)) / (size(1) - 1.f);
                // float z = (sid / size(1)) / (size(2) - 1.f);

                vec3 local;
                local(side)   = rem;
                local(side_a) = (sid % size(side_a)) / (size(side_a) - 1.f);
                local(side_b) = (sid / size(side_a)) / (size(side_b) - 1.f);

                // printf("node id %d sid %d size %d %f %f %f %d %d\n", local_id, sid, size(side_a), local(side), local(side_a), local(side_b), side_a, side_b);


                vec3 global_coords        = (local.array() * box_size.array() + node_min.array());
                vec3 global_coords_offset = global_coords + add_eps;
                int other_id              = ActiveNodeIdForGlobalPosition(tree, global_coords_offset.data(), use_quad);

                bool roi_state = true;
                if(use_roi)
                {
                    roi_state = (in_roi( tree, node_id,  roi_min, roi_max)
                    && in_roi( tree, other_id,  roi_min, roi_max) );
                }

                if (other_id != -1 && other_id != node_id 
                    && roi_state)
                {
                    // found pair
                    int out_index = atomicAdd(out_num_samples, 2);
                    CUDA_KERNEL_ASSERT(out_index <= out_global_coordinates.sizes[0]);

                    for (int d = 0; d < D; ++d)
                    {
                        out_global_coordinates(out_index + 0, d) = global_coords(d);
                        out_global_coordinates(out_index + 1, d) = global_coords(d);
                    }
                    out_ray_index(out_index + 0) = out_index + 0;
                    out_ray_index(out_index + 1) = out_index + 1;

                    out_ray_local_id(out_index + 0) = 0;
                    out_ray_local_id(out_index + 1) = 1;

                    out_node_id(out_index + 0) = node_id;
                    out_node_id(out_index + 1) = other_id;

                    out_weight(out_index + 0) = 1;
                    out_weight(out_index + 1) = 1;
                }

                // if (node_id == 0)
                // {
                //     printf("%f %f %f, %f %f %f, %d %d\n", local(0), local(1), local(2), add_eps(0), add_eps(1),
                //     add_eps(2),
                //            node_id, other_id);
                // }
            }
        }
    }
}

template <int D> 
static __global__ void NodeNeighborSamples2D(DeviceHyperTree<D> tree, Eigen::Vector<int, D> size, float epsilon,
                                            int * out_num_samples, StaticDeviceTensor<float, 2> out_global_coordinates,
                                            StaticDeviceTensor<float, 1> out_weight,
                                            StaticDeviceTensor<float, 1> out_ray_t,
                                            StaticDeviceTensor<long, 1> out_ray_index,
                                            StaticDeviceTensor<long, 1> out_ray_local_id,
                                            StaticDeviceTensor<long, 1> out_node_id,
                                            vec3 roi_min, vec3 roi_max, bool use_roi, bool use_quad, int side)
{
    int node_id = tree.active_node_ids((int)blockIdx.x);
    int local_id = threadIdx.x;

    vec3 node_min = tree.PositionMin(node_id);
    vec3 node_max = tree.PositionMax(node_id);
    vec3 box_size = node_max - node_min;
    // printf("test hre\n");
    // Eigen::Vector<int, D> out_num_samples_plane;
    // int  size_index[2] = {0, 2};
    // for(int side = 0; side < D; ++side)
    // int side = 0;
    // int side = 2;
    {    
        // int side = 2;
        Eigen::Vector<int, 2> size_index = {(side+1)%3, (side+2)%3};
        // size_index.push_back((side+1)%3);
        // size_index.push_back((side+2)%3);
        for(int tt = 0; tt < 2; ++tt)
        // int side = 1;
        {
            int side_a = size_index(tt);
            // int side_b = (side + 2) % 3;
            int side_b = size_index((tt+1)%2);
            for(int front = 0; front < 2; ++front)
            {
                vec3 add_eps(0, 0, 0);

                add_eps(side_a) = epsilon * (front ? 1 : -1);
                float rem = (front ? 1 : 0);

                for(int sid = local_id; sid < size(side_b) ; sid+= blockDim.x)
                {
                    vec3 local;
                    local(side_a)   = rem;
                    local(side_b)   = (sid % size(side_b)) / (size(side_b) - 1.f);
                    local(side)        = 0;
                    // local(side_b)   = (sid / size(side_a)) / (size(side_b) - 1.f);

                    // printf("node id %d sid %d size %d %f %f %d %d\n", local_id, sid, size(side_a), local(side_a), local(side_b), side_a, side_b);

                    vec3 global_coords      = (local.array() * box_size.array() + node_min.array());
                    // TODO to change 
                    vec3 global_coords_offset = global_coords + add_eps;

                    // printf("golobal coords %f %f %f offsets %f %f %f ", global_coords(0), global_coords(1), global_coords(2), 
                    //                                 global_coords_offset(0),global_coords_offset(1), global_coords_offset(2));
                    int other_id            = ActiveNodeIdForGlobalPosition(tree, global_coords_offset.data(), use_quad);

                    roi_min(1)              = roi_min(1) - epsilon;
                    roi_max(1)              = roi_max(1) + epsilon;

                    // printf("other id %d %d %d\n", other_id, in_roi(tree, node_id, roi_min, roi_max), in_roi(tree, other_id, roi_min, roi_max));
                    bool roi_state = true;
                    if(use_roi)
                    {
                        roi_state = (in_roi(tree, node_id, roi_min, roi_max) &&
                            in_roi(tree, other_id, roi_min, roi_max));
                    }
                    if(other_id != -1 && other_id != node_id && roi_state)
                    {
                        // found pair
                        int out_index = atomicAdd(out_num_samples, (int)2);
                        // int out_index = 0;
                        CUDA_KERNEL_ASSERT(out_index <= out_global_coordinates.sizes[0]);

                        for(int d = 0; d < D; ++d)
                        {
                            out_global_coordinates(out_index + 0, d) = global_coords(d);
                            out_global_coordinates(out_index + 1, d) = global_coords(d);
                        }
                        out_ray_index(out_index + 0) = out_index + 0;
                        out_ray_index(out_index + 1) = out_index + 1;

                        out_ray_local_id(out_index + 0 ) = 0;
                        out_ray_local_id(out_index + 1)  = 1;

                        out_node_id(out_index + 0)  = node_id;
                        out_node_id(out_index + 1) = other_id;

                        out_weight(out_index + 0) = 1;
                        out_weight(out_index + 1) = 1;
                    }
                    __syncthreads();

                }

            }
        }
    }
    // printf("out samples %d %d %d \n", out_num_samples_plane(0),out_num_samples_plane(1),out_num_samples_plane(2));
}                                            
template <int D> 
static __global__ void NodeNeighborSamplesQuad(DeviceHyperTree<D> tree, Eigen::Vector<int, D> size, float epsilon,
                                            int * out_num_samples, StaticDeviceTensor<float, 2> out_global_coordinates,
                                            StaticDeviceTensor<float, 1> out_weight,
                                            StaticDeviceTensor<float, 1> out_ray_t,
                                            StaticDeviceTensor<long, 1> out_ray_index,
                                            StaticDeviceTensor<long, 1> out_ray_local_id,
                                            StaticDeviceTensor<long, 1> out_node_id,
                                            vec3 roi_min, vec3 roi_max, bool use_roi, bool use_quad)
{
    int node_id = tree.active_node_ids((int)blockIdx.x);
    int local_id = threadIdx.x;
    vec3 node_min = tree.PositionMin(node_id);
    vec3 node_max = tree.PositionMax(node_id);
    vec3 box_size = node_max - node_min;
    int  side_ind[2] = {0, 2};
    // int side = 2;
    for(int side_index = 0; side_index < 2; ++side_index)
    {
        int side = side_ind[side_index];
        int side_a = (side + 1) % 3;
        int side_b = (side + 2) % 3;
        for(int front = 0; front < 2 ; ++ front)
        {
            vec3 add_eps(0, 0, 0);

            add_eps(side) = epsilon * (front ? 1 : -1);
            float rem     = (front ? 1 : 0);
            for(int sid = local_id; sid < size(side_a) * size(side_b); sid+= blockDim.x )
            {
                vec3 local;
                local(side)   = rem;
                local(side_a) = (sid % size(side_a))/(size(side_a) - 1.f);
                local(side_b) = (sid / size(side_a))/(size(side_b) - 1.f);
                
                vec3 global_coords          = (local.array() * box_size.array() + node_min.array());
                vec3 global_coords_offset   = global_coords + add_eps;
                // printf(" global coord %f %f %f\n ", 
                //     global_coords_offset(0),global_coords_offset(1), global_coords_offset(2)); 

                int other_id                = ActiveNodeIdForGlobalPosition(tree, global_coords_offset.data(), true);
                bool roi_state;
                if(use_roi)
                {
                    roi_state = (in_roi_quad( tree, node_id,  roi_min, roi_max,epsilon/2.f)
                    && in_roi_quad( tree, other_id,  roi_min, roi_max,epsilon/2.f) );

                }
                else
                {
                    roi_state = true;
                }
                // vec3 node_min_tmp = tree.PositionMin(node_id);
                // vec3 node_max_tmp = tree.PositionMax(node_id);

                // vec3 node_min_tmp2 = tree.PositionMin(other_id);
                // vec3 node_max_tmp2 = tree.PositionMax(other_id);
                // printf(" roi_state 1 %d roi_state 2 %d  roi0 %f %f %f %f %f %f roi1 %f %f %f %f %f %f roi2 %f %f %f %f %f %f golobal coords %f %f %f offsets %f %f %f ", 
                // in_roi_quad( tree, node_id,  roi_min, roi_max,epsilon), in_roi_quad( tree, other_id,  roi_min, roi_max,epsilon),
                //     roi_min(0), roi_min(1), roi_min(2), roi_max(0), roi_max(1), roi_max(2),
                //     node_min_tmp(0), node_min_tmp(1), node_min_tmp(2), node_max_tmp(0), node_max_tmp(1), node_max_tmp(2),
                //     node_min_tmp2(0), node_min_tmp2(1), node_min_tmp2(2), node_max_tmp2(0), node_max_tmp2(1), node_max_tmp2(2),
                //     global_coords(0), global_coords(1), global_coords(2), 
                //     global_coords_offset(0),global_coords_offset(1), global_coords_offset(2)); 
                if (other_id != -1 && other_id != node_id 
                    && roi_state)
                {
                    // found pair
                    int out_index = atomicAdd(out_num_samples, 2);
                    CUDA_KERNEL_ASSERT(out_index <= out_global_coordinates.sizes[0]);

                    for (int d = 0; d < D; ++d)
                    {
                        out_global_coordinates(out_index + 0, d) = global_coords(d);
                        out_global_coordinates(out_index + 1, d) = global_coords(d);
                    }
                    out_ray_index(out_index + 0) = out_index + 0;
                    out_ray_index(out_index + 1) = out_index + 1;

                    out_ray_local_id(out_index + 0) = 0;
                    out_ray_local_id(out_index + 1) = 1;

                    out_node_id(out_index + 0) = node_id;
                    out_node_id(out_index + 1) = other_id;

                    out_weight(out_index + 0) = 1;
                    out_weight(out_index + 1) = 1;
                }

            }
        }

    }
}
                        
SampleList HyperTreeBaseImpl::NodeNeighborSamples(Eigen::Vector<int, -1> size, double epsilon , vec3 roi_min, vec3 roi_max, bool use_roi, bool use_quad = false)
{
    int predicted_samples = NumActiveNodes() * 6 * size(0) * size(0) * 2;
    // if(use_quad)
    // {
    //     predicted_samples = NumActiveNodes() * 6 * 2 * size(0) * 2;
    // }

    SampleList list;
    list.Allocate(predicted_samples, D(), node_position_min.device());
    auto out_num_samples = torch::zeros({2}, node_position_min.options().dtype(torch::kInt32));

    // Eigen::Vector<int, -1> out_num_samples_plane;
    // auto out_num_samples_plane = torch::zeros({3}, node_position_min.options().dtype(torch::kInt32));
    // bool use_roi = params->octree_params.tree_optimizer_params.use_tree_roi;
    switch (D())
    {
        case 3:
            // if(use_quad)
            // {
            //     // printf("use quad tree\n");
            //     ::NodeNeighborSamples2D<3><<<NumActiveNodes(), 128>>>(this, size, epsilon, out_num_samples.data_ptr<int>(),
            //                                                         list.global_coordinate, list.weight, list.ray_t,
            //                                                         list.ray_index, list.local_index_in_ray, list.node_id, 
            //                                                         roi_min, roi_max, use_roi,use_quad, out_num_samples_plane.data_ptr<int>());
            // }
            // else
            {
                ::NodeNeighborSamples<3><<<NumActiveNodes(), 128>>>(this, size, epsilon, out_num_samples.data_ptr<int>(),
                                                                    list.global_coordinate, list.weight, list.ray_t,
                                                                    list.ray_index, list.local_index_in_ray, list.node_id, 
                                                                    roi_min, roi_max, use_roi,use_quad);
            }

            break;
        default:
            CHECK(false);
    }
    // printf("out samples %d %d %d \n", out_num_samples_plane(0),out_num_samples_plane(1),out_num_samples_plane(2));

    // out_num_samples    = out_num_samples.cpu();
    // int actual_samples = out_num_samples.data_ptr<int>()[0];

    // int actual_samples2 = out_num_samples[0].item().toInt();
    // int actual_samples3 = out_num_samples[1].item().toInt();
    // std::cout << "actual samples " << actual_samples << " " << actual_samples2 <<" " << actual_samples3<< std::endl;

    int actual_samples = out_num_samples[0].item().toInt();
    // std::cout <<"neighbor samples " << actual_samples << " " << predicted_samples << std::endl;
    // // std::cout << "after sample per " << out_num_samples_plane.data_ptr<int>()[0]
    // //         << " " << out_num_samples_plane.data_ptr<int>()[1]
    // //         << " " << out_num_samples_plane.data_ptr<int>()[2] << std::endl;
    // out_num_samples_plane = out_num_samples_plane.cpu();
    // int * num_plane = out_num_samples_plane.data_ptr<int>();
    // std::cout << "after sample " ;
    // std::cout << num_plane[0] << " " << num_plane[1] << " " << num_plane[2] << std::endl;
    // PrintTensorInfo(list.global_coordinate);
    // char c = getchar();
    list.Shrink(actual_samples);
    CHECK_LE(actual_samples, predicted_samples);

    CUDA_SYNC_CHECK_ERROR();
    return list;
}


SampleList HyperTreeBaseImpl::NodeNeighborSamplesPlane2D(Eigen::Vector<int, -1> size, double epsilon , int plane_index, vec3 roi_min,
                                    vec3 roi_max, bool use_roi, bool use_quad = false)
{
    int size_max = size.maxCoeff();
    int predicted_samples = NumActiveNodes() * 6 * size_max * size_max * 2;
    int max_value = std::max(std::max(size(0), size(1)), size(2));
    // printf("max value is %d\n", max_value);
    if(use_quad)
    {
        predicted_samples = NumActiveNodes() * 4 * max_value * 2;
    }

    // SampleList * list = new SampleList[3];
    SampleList list;
    // for(int i = 0; i < 3; ++i)
    {
        list.Allocate(predicted_samples, D(), node_position_min.device());
        auto out_num_samples = torch::zeros({2}, node_position_min.options().dtype(torch::kInt32));

        // PrintTensorInfo(node_position_min);
        // PrintTensorInfo(list.global_coordinate);
        // int out_num_samples1[2];
        // Eigen::Vector<int, -1> out_num_samples_plane;
        // bool use_roi = params->octree_params.tree_optimizer_params.use_tree_roi;
        // printf("test here 0\n");
        cudaDeviceSynchronize();
        // printf("test here 1\n");
        switch (D())
        {
            case 3:
                if(use_quad)
                {
                    printf("use quad tree test \n");
                    ::NodeNeighborSamples2D<3><<<NumActiveNodes(), 128>>>(this, size, epsilon, out_num_samples.data_ptr<int>(),
                                                                        list.global_coordinate, list.weight, list.ray_t,
                                                                        list.ray_index, list.local_index_in_ray, list.node_id, 
                                                                        roi_min, roi_max, use_roi,use_quad, plane_index);
                }
                else
                {
                    ::NodeNeighborSamplesQuad<3><<<NumActiveNodes(), 128>>>(this, size, epsilon, out_num_samples.data_ptr<int>(),
                                                                        list.global_coordinate, list.weight, list.ray_t,
                                                                        list.ray_index, list.local_index_in_ray, list.node_id, 
                                                                        roi_min, roi_max, use_roi,use_quad);
                }

                break;
            default:
                CHECK(false);
        }
        cudaDeviceSynchronize();
        // printf("out samples %d %d %d \n", out_num_samples_plane(0),out_num_samples_plane(1),out_num_samples_plane(2));

    //     // out_num_samples    = out_num_samples.cpu();
    //     // int actual_samples1 = out_num_samples.data_ptr<int>()[0];
    //     int actual_samples1 = out_num_samples1[0];

    // // int actual_samples2 = out_num_samples[0].item().toInt();
    // // int actual_samples3 = out_num_samples[1].item().toInt();
    // // std::cout << "actual samples " << actual_samples << " " << actual_samples2 <<" " << actual_samples3<< std::endl;
    //     printf("test here 0\n");
    //     std::cout << "actual_samples1 " << actual_samples1 << "predicted_samples " << predicted_samples << std::endl;
        // PrintTensorInfo(list.global_coordinate);
        // PrintTensorInfo(out_num_samples);

        int actual_samples = out_num_samples[0].item().toInt();
        // std::cout << "acutal samples " << actual_samples << "predicted_samples " << predicted_samples << std::endl;

        // std::cout <<"neighbor samples " << actual_samples << " " << predicted_samples << std::endl;
        // std::cout << "after sample per " << out_num_samples_plane.data_ptr<int>()[0]
        //         << " " << out_num_samples_plane.data_ptr<int>()[1]
        //         << " " << out_num_samples_plane.data_ptr<int>()[2] << std::endl;
        // PrintTensorInfo(list.global_coordinate);
        // char c = getchar();
        list.Shrink(actual_samples);
        // PrintTensorInfo(list[i].global_coordinate);
        CHECK_LE(actual_samples, predicted_samples);

    }


    CUDA_SYNC_CHECK_ERROR();
    return list;
}
