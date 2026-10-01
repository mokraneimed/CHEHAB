#pragma once

#include "fheco/dsl/plaintext_matrix.hpp"

#include <vector>

namespace fheco
{
Tensor matmul(const Tensor &input, const PlaintextMatrix &matrix);
Tensor poly_activate(const Tensor &input, const std::vector<double> &coefficients);
} // namespace fheco
