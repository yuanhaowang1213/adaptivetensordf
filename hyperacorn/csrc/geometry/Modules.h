#pragma once
#include "saiga/core/image/image.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/torch/TorchHelper.h"

#include <torch/torch.h>


class ScaledSinImpl : public torch::nn::Module
{
   public:
    ScaledSinImpl(float w) : w(w) {}

    at::Tensor forward(at::Tensor x) { return torch::sin(x * w); }
    float w;
};

TORCH_MODULE(ScaledSin);



class FCBlockImpl : public torch::nn::Module
{
   public:
    FCBlockImpl(int in_features, int out_features, int num_hidden_layers, int hidden_features,
                bool outermost_linear = true, std::string non_linearity = "relu" , bool add_drop_out = false)
        : in_features(in_features), out_features(out_features)
    {
        auto make_lin = [](int in, int out)
        {
            auto lin = torch::nn::Linear(in, out);
            torch::nn::init::kaiming_normal_(lin->weight, 0, torch::kFanIn, torch::kReLU);
            std::cout << "(lin " << in << "->" << out << ") ";
            return lin;
        };

        if(num_hidden_layers < 1)
        {
            seq->push_back(make_lin(in_features, out_features));
            if (!outermost_linear)
            {
                seq->push_back(Saiga::ActivationFromString(non_linearity));
                std::cout << "(" << non_linearity << ") ";
            }
            // add drop out
            if(add_drop_out)
            {
                printf("add drop out\n");
                seq->push_back(torch::nn::Dropout(0.2));
            }
            register_module("seq", seq);
        }
        // else if(num_hidden_layers == 1)
        // {
        //     seq->push_back(make_lin(in_features, hidden_features));
        //     seq->push_back(torch::nn::Functional(torch::leaky_relu,0.01));
        //     seq->push_back(make_lin(hidden_features, out_features));
        //     // seq->push_back(Saiga::ActivationFromString(lea))
        //     if (!outermost_linear)
        //     {
        //         seq->push_back(Saiga::ActivationFromString(non_linearity));
        //         std::cout << "(" << non_linearity << ") ";
                
        //     }
        //     register_module("seq",seq);
        // }
        else
        {
            seq->push_back(make_lin(in_features, hidden_features));
            if(non_linearity == "leakyrelu")
            {
                seq->push_back(torch::nn::Functional(torch::leaky_relu, 0.01));
            }
            else
            {
                seq->push_back(Saiga::ActivationFromString(non_linearity));
            }
            std::cout << "(" << non_linearity << ") ";

            for (int i = 0; i < num_hidden_layers; ++i)
            {
                seq->push_back(make_lin(hidden_features, hidden_features));
                if(non_linearity == "leakyrelu")
                {
                    seq->push_back(torch::nn::Functional(torch::leaky_relu, 0.01));
                }
                else
                {
                    seq->push_back(Saiga::ActivationFromString(non_linearity));
                }
                std::cout << "(" << non_linearity << ") ";
            }

            seq->push_back(make_lin(hidden_features, out_features));
            if (!outermost_linear)
            {
                seq->push_back(Saiga::ActivationFromString(non_linearity));
                std::cout << "(" << non_linearity << ") ";
            }
            // add drop out
            if(add_drop_out)
            {
                printf("add drop out\n");
                seq->push_back(torch::nn::Dropout(0.2));
            }
            register_module("seq", seq);

        }


        int num_params = 0;
        for (auto& t : this->parameters())
        {
            num_params += t.numel();
        }
        std::cout << "  |  #Params " << num_params;
        std::cout << std::endl;
    }

    void reset_module_parameters()
    {
        int count = 0;
        for(auto& module : modules(/*include_self=*/false))
        {
            if(auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get()))
            {
                // printf("init a linear %d\n",count);
                // torch::nn::init::normal_(M->weight, 0, 0.01);
                // torch::nn::init::constant_(M->bias, 0);
                M->reset_parameters();
                count +=1;
            }
        }
        printf("reset FCB %d\n", count);
    }
    at::Tensor forward(at::Tensor x)
    {
        CHECK_EQ(in_features, x.size(-1));
        x = seq->forward(x);
        // printf("decoder test \n");
        // PrintTensorInfo(x);
        CHECK_EQ(out_features, x.size(-1));
        return x;
    }

    int in_features, out_features;
    torch::nn::Sequential seq;
};

TORCH_MODULE(FCBlock);
