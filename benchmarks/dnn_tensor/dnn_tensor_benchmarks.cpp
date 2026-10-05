#include "fheco/fheco.hpp"
#include "fheco/ckks/ckks_params.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;
using namespace fheco;

namespace
{
constexpr int kDefaultSlots = 256;

vector<vector<double>> kernel3()
{
  return {{-1.0, -1.0, -1.0}, {-1.0, 8.0, -1.0}, {-1.0, -1.0, -1.0}};
}

vector<vector<double>> dense_weights(int outputs, int inputs)
{
  vector<vector<double>> weights(outputs, vector<double>(inputs));
  for (int output = 0; output < outputs; ++output)
    for (int input = 0; input < inputs; ++input)
      weights[output][input] = static_cast<double>((output + input) % 3 - 1);
  return weights;
}

vector<vector<vector<vector<double>>>> conv_weights(int output_channels, int input_channels, int kernel_size)
{
  return vector<vector<vector<vector<double>>>>(
    output_channels, vector<vector<vector<double>>>(
      input_channels, vector<vector<double>>(kernel_size, vector<double>(kernel_size, 1.0))));
}


void build_conv2d(int slots)
{
  Tensor image("image", TensorShape{1, 16, 16});
  PlaintextMatrix::Conv2dParams params{16, 16, 3};
  matmul(image, PlaintextMatrix::from_conv2d(kernel3(), params, slots)).set_output("conv_output");
}

void build_full_dnn(int slots)
{
  Tensor image("image", TensorShape{1, 16, 16});
  PlaintextMatrix::Conv2dParams conv1{16, 16, 3};
  Tensor layer1 = poly_activate(matmul(image, PlaintextMatrix::from_conv2d(kernel3(), conv1, slots)), {0, 0, 1});
  PlaintextMatrix::PoolParams pool{14, 14, 2, 2};
  Tensor layer2 = matmul(layer1, PlaintextMatrix::from_pool(pool, slots));
  PlaintextMatrix::Conv2dParams conv2{7, 7, 3};
  Tensor layer3 = poly_activate(matmul(layer2, PlaintextMatrix::from_conv2d(kernel3(), conv2, slots)), {0, 0, 1});
  matmul(layer3, PlaintextMatrix::from_dense(dense_weights(10, 25), slots)).set_output("classification");
}

void build_cryptonets(int slots)
{
  Tensor image("image", TensorShape{1, 16, 16}, PackingLayout::INTERLEAVED_HWC);
  PlaintextMatrix::Conv2dParams conv1{16, 16, 3, 2, 0, 1, 3, PackingLayout::INTERLEAVED_HWC};
  Tensor layer1 = poly_activate(matmul(image, PlaintextMatrix::from_conv2d(conv_weights(3, 1, 3), conv1, slots)), {0, 0, 1});
  PlaintextMatrix::PoolParams pool{7, 7, 2, 2, 3, PackingLayout::INTERLEAVED_HWC};
  Tensor layer2 = matmul(layer1, PlaintextMatrix::from_pool(pool, slots));
  PlaintextMatrix::Conv2dParams conv2{3, 3, 3, 1, 0, 3, 4, PackingLayout::INTERLEAVED_HWC};
  Tensor layer3 = poly_activate(matmul(layer2, PlaintextMatrix::from_conv2d(conv_weights(4, 3, 3), conv2, slots)), {0, 0, 1});
  matmul(layer3, PlaintextMatrix::from_dense(dense_weights(10, 4), slots, PackingLayout::INTERLEAVED_HWC)).set_output("classification");
}



//LoLA: Low-Latency CryptoNet from Figure 3 of https://arxiv.org/pdf/1812.10659

void build_lola(int slots)
{
  Tensor image("image", TensorShape{1, 28, 28});
  PlaintextMatrix::Conv2dParams conv1_params{28, 28, 5, 2, 0, 1, 5};
  Tensor conv1 = matmul(image, PlaintextMatrix::from_conv2d(conv_weights(5, 1, 5), conv1_params, slots));
  Tensor act1 = poly_activate(conv1, {0, 0, 1});

  // Dense 1: 720 -> 100
  Tensor dense1 = matmul(act1, PlaintextMatrix::from_dense(dense_weights(100, 720), slots));
  Tensor act2 = poly_activate(dense1, {0, 0, 1});

  // Dense 2: 100 -> 10
  matmul(act2, PlaintextMatrix::from_dense(dense_weights(10, 100), slots)).set_output("classification");
}


// MLP: Multi-Layer Perceptron from SecureML (https://eprint.iacr.org/2017/396.pdf)
 
void build_mlp(int slots)
{
  Tensor input("input", TensorShape{1, 1, 784});
  Tensor layer1 = poly_activate(matmul(input, PlaintextMatrix::from_dense(dense_weights(128, 784), slots)), {0, 0, 1});
  Tensor layer2 = poly_activate(matmul(layer1, PlaintextMatrix::from_dense(dense_weights(128, 128), slots)), {0, 0, 1});
  matmul(layer2, PlaintextMatrix::from_dense(dense_weights(10, 128), slots)).set_output("classification");
}


 // ResNet: Residual Network from Orion Parameter Set

void build_resnet(int slots)
{
  Tensor image("image", TensorShape{3, 16, 16});

  // Initial convolution
  PlaintextMatrix::Conv2dParams conv1_params{16, 16, 3, 1, 1, 3, 8};
  Tensor conv1 = poly_activate(matmul(image, PlaintextMatrix::from_conv2d(conv_weights(8, 3, 3), conv1_params, slots)), {0, 0, 1});

  // Residual Block
  PlaintextMatrix::Conv2dParams r1_params{16, 16, 3, 1, 1, 8, 8};
  Tensor r1 = poly_activate(matmul(conv1, PlaintextMatrix::from_conv2d(conv_weights(8, 8, 3), r1_params, slots)), {0, 0, 1});
  PlaintextMatrix::Conv2dParams r2_params{16, 16, 3, 1, 1, 8, 8};
  Tensor r2 = matmul(r1, PlaintextMatrix::from_conv2d(conv_weights(8, 8, 3), r2_params, slots));

  // Skip connection addition (r2 + conv1) and post-activation
  Tensor res_out = poly_activate(r2 + conv1, {0, 0, 1});

  // Downsampling stage
  PlaintextMatrix::PoolParams pool_params{16, 16, 2, 2, 8};
  Tensor pooled = matmul(res_out, PlaintextMatrix::from_pool(pool_params, slots));

  PlaintextMatrix::Conv2dParams conv2_params{8, 8, 3, 1, 1, 8, 16};
  Tensor conv2 = poly_activate(matmul(pooled, PlaintextMatrix::from_conv2d(conv_weights(16, 8, 3), conv2_params, slots)), {0, 0, 1});

  // Global average pool
  PlaintextMatrix::PoolParams gap_params{8, 8, 8, 8, 16};
  Tensor gap = matmul(conv2, PlaintextMatrix::from_pool(gap_params, slots));

  // Dense classification
  matmul(gap, PlaintextMatrix::from_dense(dense_weights(10, 16), slots)).set_output("classification");
}

// AlexNet: 5 Convolutional layers + Pooling + 2 Fully Connected layers

void build_alexnet(int slots)
{
  Tensor image("image", TensorShape{3, 16, 16});

  // Conv1 + Pool1
  PlaintextMatrix::Conv2dParams c1{16, 16, 3, 1, 1, 3, 8};
  Tensor c1_out = poly_activate(matmul(image, PlaintextMatrix::from_conv2d(conv_weights(8, 3, 3), c1, slots)), {0, 0, 1});
  PlaintextMatrix::PoolParams p1{16, 16, 2, 2, 8};
  Tensor p1_out = matmul(c1_out, PlaintextMatrix::from_pool(p1, slots));

  // Conv2 + Pool2
  PlaintextMatrix::Conv2dParams c2{8, 8, 3, 1, 1, 8, 16};
  Tensor c2_out = poly_activate(matmul(p1_out, PlaintextMatrix::from_conv2d(conv_weights(16, 8, 3), c2, slots)), {0, 0, 1});
  PlaintextMatrix::PoolParams p2{8, 8, 2, 2, 16};
  Tensor p2_out = matmul(c2_out, PlaintextMatrix::from_pool(p2, slots));

  // Conv3, Conv4, Conv5
  PlaintextMatrix::Conv2dParams c3{4, 4, 3, 1, 1, 16, 16};
  Tensor c3_out = poly_activate(matmul(p2_out, PlaintextMatrix::from_conv2d(conv_weights(16, 16, 3), c3, slots)), {0, 0, 1});

  PlaintextMatrix::Conv2dParams c4{4, 4, 3, 1, 1, 16, 16};
  Tensor c4_out = poly_activate(matmul(c3_out, PlaintextMatrix::from_conv2d(conv_weights(16, 16, 3), c4, slots)), {0, 0, 1});

  PlaintextMatrix::Conv2dParams c5{4, 4, 3, 1, 1, 16, 8};
  Tensor c5_out = poly_activate(matmul(c4_out, PlaintextMatrix::from_conv2d(conv_weights(8, 16, 3), c5, slots)), {0, 0, 1});

  // Pool3
  PlaintextMatrix::PoolParams p3{4, 4, 2, 2, 8};
  Tensor p3_out = matmul(c5_out, PlaintextMatrix::from_pool(p3, slots));

  // FC1 + FC2
  Tensor fc1 = poly_activate(matmul(p3_out, PlaintextMatrix::from_dense(dense_weights(64, 32), slots)), {0, 0, 1});
  matmul(fc1, PlaintextMatrix::from_dense(dense_weights(10, 64), slots)).set_output("classification");
}

void build(const string &benchmark, int slots)
{
  if (benchmark == "conv2d") build_conv2d(slots);
  else if (benchmark == "full_dnn") build_full_dnn(slots);
  else if (benchmark == "cryptonets") build_cryptonets(slots);
  else if (benchmark == "lola") build_lola(slots);
  else if (benchmark == "mlp") build_mlp(slots);
  else if (benchmark == "resnet") build_resnet(slots);
  else if (benchmark == "alexnet") build_alexnet(slots);
  else throw invalid_argument("--benchmark must be conv2d, full_dnn, cryptonets, lola, mlp, resnet, or alexnet");
}

int get_default_slots(const string &benchmark)
{
  if (benchmark == "lola" || benchmark == "mlp") return 4096;
  if (benchmark == "resnet") return 32768;
  if (benchmark == "alexnet") return 16384;
  return kDefaultSlots;
}

bool has_orion_ckks_params(const string &benchmark)
{
  return benchmark == "lola" || benchmark == "mlp" || benchmark == "resnet" || benchmark == "alexnet";
}

ckks::CKKSParams get_orion_ckks_params(const string &benchmark)
{
  ckks::CKKSParams params;
  if (benchmark == "lola" || benchmark == "mlp")
  {
    // Config for LoLA (Figure 3, arXiv:1812.10659) & MLP (eprint.iacr.org/2017/396)
    params.log_n = 13;
    params.log_q = {29, 26, 26, 26, 26, 26};
    params.log_p = {29, 29};
    params.log_scale = 26;
    params.hamming_weight = 8192;
    params.ring_type = ckks::RingType::ConjugateInvariant;
    params.enable_bootstrap = false;
  }
  else if (benchmark == "resnet")
  {
    // Config for ResNet Parameter Set
    params.log_n = 16;
    params.log_q = {55, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40};
    params.log_p = {61, 61, 61};
    params.log_scale = 40;
    params.hamming_weight = 192;
    params.ring_type = ckks::RingType::Standard;
    params.log_p_boot = {61, 61, 61, 61, 61, 61, 61, 61};
    params.enable_bootstrap = true;
  }
  else if (benchmark == "alexnet")
  {
    // Config for AlexNet
    params.log_n = 15;
    params.log_q = {55, 40, 40, 40, 40, 40, 40, 40, 40};
    params.log_p = {61, 61, 61};
    params.log_scale = 40;
    params.hamming_weight = 192;
    params.ring_type = ckks::RingType::Standard;
    params.enable_bootstrap = false;
  }
  return params;
}

} // namespace

int main(int argc, char **argv)
{
  string benchmark = "cryptonets";
  bool global_bsgs = false;
  bool optimize = true;
  int user_slots = -1;

  for (int index = 1; index < argc; ++index)
  {
    const string argument = argv[index];
    if (argument == "--benchmark" && ++index < argc) benchmark = argv[index];
    else if (argument == "--slots" && ++index < argc) user_slots = stoi(argv[index]);
    else if (argument == "--bsgs-base" && ++index < argc)
    {
      const string value = argv[index];
      if (value == "global") global_bsgs = true;
      else Compiler::set_canonical_bsgs_base(stoi(value));
    }
    else if (argument == "--optimizer" && ++index < argc)
    {
      const string value = argv[index];
      if (value == "none") optimize = false;
      else if (value != "trs") throw invalid_argument("--optimizer must be trs or none");
    }
    else throw invalid_argument("unknown argument: " + argument);
  }

  const int slots = user_slots > 0 ? user_slots : get_default_slots(benchmark);

  Compiler::enable_cse();
  Compiler::enable_order_operands();
  Compiler::enable_const_folding();
  const string function_name = "tensor_" + benchmark;
  if (global_bsgs)
  {
    Compiler::clear_diag_collector();
    Compiler::create_func(function_name, slots, 20, false, true);
    build(benchmark, slots);
    Compiler::set_canonical_bsgs_base(Compiler::compute_global_bsgs_base());
    Compiler::clear_all_funcs();
    Compiler::clear_diag_collector();
  }

  const auto &func = Compiler::create_func(function_name, slots, 20, false, true);
  const auto started = chrono::steady_clock::now();
  build(benchmark, slots);
  if (optimize) Compiler::compile(func, Compiler::Ruleset::depth, trs::RewriteHeuristic::bottom_up);

  filesystem::create_directories("he");
  const string output = "he/" + function_name + "_auto.go";
  ofstream generated(output);
  if (!generated) throw runtime_error("cannot create " + output);

  if (has_orion_ckks_params(benchmark))
  {
    ckks::CKKSParams params = get_orion_ckks_params(benchmark);
    Compiler::gen_lattigo_code(func, generated, params);
  }
  else
  {
    Compiler::gen_lattigo_code(func, generated);
  }

  cout << "Tensor benchmark: " << benchmark << '\n'
       << "SIMD slots: " << slots << '\n'
       << "DSL generation time: "
       << chrono::duration<double, milli>(chrono::steady_clock::now() - started).count() << " ms\n"
       << "Generated: " << output << endl;
}
