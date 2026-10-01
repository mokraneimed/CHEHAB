#include "fheco/dsl/plaintext_matrix.hpp"

#include <stdexcept>

using namespace std;

namespace fheco
{
namespace
{
void validate_common(int height, int width, int kernel, int stride, int channels, int slots)
{
  if (height <= 0 || width <= 0 || kernel <= 0 || stride <= 0 || channels <= 0 || slots <= 0)
    throw invalid_argument("Tensor matrix dimensions, stride, channels, and slots must be positive");
}
}

PlaintextMatrix PlaintextMatrix::from_conv2d(
  const vector<vector<double>> &kernel, Conv2dParams params, int total_slots)
{
  if (kernel.size() != static_cast<size_t>(params.kernel_size) || kernel.empty() ||
      kernel.front().size() != static_cast<size_t>(params.kernel_size))
    throw invalid_argument("kernel dimensions must match kernel_size");
  vector<vector<vector<vector<double>>>> expanded(
    params.channels_out, vector<vector<vector<double>>>(params.channels_in, kernel));
  return from_conv2d(expanded, params, total_slots);
}

PlaintextMatrix PlaintextMatrix::from_conv2d(
  const vector<vector<vector<vector<double>>>> &kernel, Conv2dParams params, int total_slots)
{
  validate_common(params.image_height, params.image_width, params.kernel_size, params.stride,
                  params.channels_in, total_slots);
  if (params.channels_out <= 0 || params.padding < 0 ||
      kernel.size() != static_cast<size_t>(params.channels_out))
    throw invalid_argument("invalid convolution parameters or kernel output channels");
  for (const auto &out_channel : kernel)
    if (out_channel.size() != static_cast<size_t>(params.channels_in))
      throw invalid_argument("kernel input channels must match channels_in");

  const int h_out = (params.image_height + 2 * params.padding - params.kernel_size) / params.stride + 1;
  const int w_out = (params.image_width + 2 * params.padding - params.kernel_size) / params.stride + 1;
  if (h_out <= 0 || w_out <= 0)
    throw invalid_argument("convolution kernel does not fit input");

  unordered_map<int, vector<double>> diagonals;
  const auto in_slot = [&](int channel, int row, int col) {
    return params.layout == PackingLayout::ROW_MAJOR
      ? (channel * params.image_height + row) * params.image_width + col
      : (row * params.image_width + col) * params.channels_in + channel;
  };
  const auto out_slot = [&](int channel, int row, int col) {
    return params.layout == PackingLayout::ROW_MAJOR
      ? (channel * h_out + row) * w_out + col
      : (row * w_out + col) * params.channels_out + channel;
  };

  for (int row_out = 0; row_out < h_out; ++row_out)
    for (int col_out = 0; col_out < w_out; ++col_out)
      for (int out_channel = 0; out_channel < params.channels_out; ++out_channel)
      {
        const int output = out_slot(out_channel, row_out, col_out);
        for (int kernel_row = 0; kernel_row < params.kernel_size; ++kernel_row)
          for (int kernel_col = 0; kernel_col < params.kernel_size; ++kernel_col)
          {
            const int row_in = row_out * params.stride + kernel_row - params.padding;
            const int col_in = col_out * params.stride + kernel_col - params.padding;
            if (row_in < 0 || row_in >= params.image_height || col_in < 0 || col_in >= params.image_width)
              continue;
            for (int input_channel = 0; input_channel < params.channels_in; ++input_channel)
            {
              const auto &plane = kernel[out_channel][input_channel];
              if (plane.size() != static_cast<size_t>(params.kernel_size) ||
                  plane[kernel_row].size() != static_cast<size_t>(params.kernel_size))
                throw invalid_argument("kernel dimensions must match kernel_size");
              const int input = in_slot(input_channel, row_in, col_in);
              const int diagonal = ((input - output) % total_slots + total_slots) % total_slots;
              auto &values = diagonals[diagonal];
              if (values.empty()) values.assign(total_slots, 0.0);
              values[output] += plane[kernel_row][kernel_col];
            }
          }
      }

  return PlaintextMatrix(move(diagonals), total_slots,
                         TensorShape{static_cast<size_t>(params.channels_out),
                                     static_cast<size_t>(h_out), static_cast<size_t>(w_out)},
                         params.layout);
}

PlaintextMatrix PlaintextMatrix::from_pool(PoolParams params, int total_slots)
{
  const int stride = params.stride == 0 ? params.kernel_size : params.stride;
  validate_common(params.input_height, params.input_width, params.kernel_size, stride,
                  params.channels, total_slots);
  const int h_out = (params.input_height - params.kernel_size) / stride + 1;
  const int w_out = (params.input_width - params.kernel_size) / stride + 1;
  if (h_out <= 0 || w_out <= 0) throw invalid_argument("pooling kernel does not fit input");

  unordered_map<int, vector<double>> diagonals;
  const auto in_slot = [&](int channel, int row, int col) {
    return params.layout == PackingLayout::ROW_MAJOR
      ? (channel * params.input_height + row) * params.input_width + col
      : (row * params.input_width + col) * params.channels + channel;
  };
  const auto out_slot = [&](int channel, int row, int col) {
    return params.layout == PackingLayout::ROW_MAJOR
      ? (channel * h_out + row) * w_out + col
      : (row * w_out + col) * params.channels + channel;
  };
  const double scale = 1.0 / (params.kernel_size * params.kernel_size);
  for (int row_out = 0; row_out < h_out; ++row_out)
    for (int col_out = 0; col_out < w_out; ++col_out)
      for (int channel = 0; channel < params.channels; ++channel)
      {
        const int output = out_slot(channel, row_out, col_out);
        for (int kernel_row = 0; kernel_row < params.kernel_size; ++kernel_row)
          for (int kernel_col = 0; kernel_col < params.kernel_size; ++kernel_col)
          {
            const int input = in_slot(channel, row_out * stride + kernel_row, col_out * stride + kernel_col);
            const int diagonal = ((input - output) % total_slots + total_slots) % total_slots;
            auto &values = diagonals[diagonal];
            if (values.empty()) values.assign(total_slots, 0.0);
            values[output] += scale;
          }
      }
  return PlaintextMatrix(move(diagonals), total_slots,
                         TensorShape{static_cast<size_t>(params.channels),
                                     static_cast<size_t>(h_out), static_cast<size_t>(w_out)},
                         params.layout);
}

PlaintextMatrix PlaintextMatrix::from_dense(
  const vector<vector<double>> &weights, int total_slots, PackingLayout layout)
{
  if (weights.empty() || weights.front().empty() || total_slots <= 0)
    throw invalid_argument("dense weights and slots must not be empty");
  const int input_size = static_cast<int>(weights.front().size());
  unordered_map<int, vector<double>> diagonals;
  for (int output = 0; output < static_cast<int>(weights.size()); ++output)
  {
    if (static_cast<int>(weights[output].size()) != input_size)
      throw invalid_argument("dense weight rows must have equal size");
    for (int input = 0; input < input_size; ++input)
    {
      if (weights[output][input] == 0.0) continue;
      const int diagonal = ((input - output) % total_slots + total_slots) % total_slots;
      auto &values = diagonals[diagonal];
      if (values.empty()) values.assign(total_slots, 0.0);
      values[output] += weights[output][input];
    }
  }
  return PlaintextMatrix(move(diagonals), total_slots,
                         TensorShape{1, 1, weights.size()}, layout);
}
} // namespace fheco
