#include "fhecompiler/fhecompiler.hpp"

inline fhecompiler::Ciphertext sum_all_slots(fhecompiler::Ciphertext &x, int vector_size)
{
  fhecompiler::Ciphertext result = x;
  int step = vector_size - 1;
  if (step < 0)
    throw std::logic_error("negative rotation step");
  fhecompiler::Ciphertext rots_sum = x << step;
  step--;
  for (; step > 0;)
  {
    rots_sum += (x << (step--));
  }
  // result of sum will be in the first slot
  result += rots_sum;
  return result;
}

inline fhecompiler::Ciphertext sum_all_slots2(const fhecompiler::Ciphertext &x, size_t vector_size)
{
  fhecompiler::Ciphertext result = x;

  auto clog2 = [](int32_t x) -> int32_t {
    int32_t r = 0;
    while (x > 1)
    {
      x /= 2;
      r += 1;
    }
    return r;
  };

  /*
    A : 8192 x 64
    B : 64 x 256
    n = poly_modulus_degree = 8192/64 = 128

    1 1st 128 lines
    1 2nd 128 lines

    n = 8192

    L is a line of A', L = [1,2,4,5|6,4,3,2|.....] sum of segments

    A'(A packed) : 64 x 64
    B = 64 x 256
    C = A' x B
  */

  auto next_power_of_2 = [&clog2](size_t n) -> size_t {
    if (__builtin_popcount(n) == 1)
      return n;

    auto log2_of_n = clog2(n);

    // I assume no overflow as n will be in range [2^10, 2^16]
    return (1 << (log2_of_n + 1));
  };

  vector_size = next_power_of_2(vector_size);
  int32_t max_num_steps = clog2(vector_size);

  int32_t rot_step = 1;
  for (; max_num_steps--; rot_step *= 2)
  {
    fhecompiler::Ciphertext new_rotated_cipher = result << rot_step;
    result += new_rotated_cipher;
  }
  // result of sum will be in the first slot
  return result;
}

int main()
{

  try
  {
    fhecompiler::init("matrix_mul", 40);

    size_t polynomial_modulus_degree = 4096;
    size_t plaintext_modulus = 786433;

    std::vector<std::vector<int64_t>> A = {{1, 2, 3, -2}, {-5, 3, 2, 0}, {1, 0, 1, -3}, {5, 3, 2, 0}, {5, 3, 2, 0}};
    std::vector<std::vector<int64_t>> B = {{0, 1, 9}, {-7, -10, 2}, {1, 9, 0}, {-8, 2, 18}};

    const int N = 64;
    const int M = 64;
    const int P = 64;
    const int Q = 256;

    /*
    for (size_t i = 0; i < N; i++)
    {
      std::vector<int64_t> line;
      for (size_t j = 0; j < M; j++)
      {
        std::vector<int64_t> line;
        for (size_t j = 0; j < M; j++)
        {
          line.push_back((i + 1) * (j + 1));
        }
        A.push_back(line);
      }
      A.push_back(line);
    }
    for (size_t i = 0; i < P; i++)
    {
      std::vector<int64_t> line;
      for (size_t j = 0; j < Q; j++)
      {
        std::vector<int64_t> line;
        for (size_t j = 0; j < Q; j++)
        {
          line.push_back((i + 1) * (j + 1));
        }
        B.push_back(line);
      }

      */

    std::vector<fhecompiler::Ciphertext> A_encrypted;
    // encrypt by line for matrix A
    for (std::vector<int64_t> line : A)
    {
      fhecompiler::Ciphertext line_encrypted = fhecompiler::Ciphertext::encrypt(line);
      A_encrypted.push_back(line_encrypted);
    }
    // encrypt by column for matrix B
    std::vector<fhecompiler::Ciphertext> B_encrypted;
    for (size_t column_index = 0; column_index < B[0].size(); column_index++)
    {
      std::vector<int64_t> column_data;
      for (size_t i = 0; i < B.size(); i++)
      {
        column_data.push_back(B[i][column_index]);
      }
      fhecompiler::Ciphertext column_encrypted = fhecompiler::Ciphertext::encrypt(column_data);
      B_encrypted.push_back(column_encrypted);
    }

    // C contains result of multiplication
    std::vector<fhecompiler::Ciphertext> C_encrypted;
    // make outputs
    std::vector<fhecompiler::Ciphertext> outputs;
    for (size_t i = 0; i < A_encrypted.size(); i++)
    {
      std::vector<fhecompiler::Ciphertext> temp_ciphers;
      // A.size() pt-ct multiplications
      // A.size() copying
      for (size_t j = 0; j < B_encrypted.size(); j++)
      {
        std::vector<int64_t> mask(A[0].size(), 0);
        mask[0] = 1;
        fhecompiler::Ciphertext simd_product = A_encrypted[i] * B_encrypted[j];
        fhecompiler::Ciphertext temp_cipher = sum_all_slots(simd_product, A[0].size()) * mask;
        if (j > 0)
          temp_cipher >>= j;
        temp_ciphers.push_back(temp_cipher);
      }
      fhecompiler::Ciphertext c_line = temp_ciphers[0];
      for (size_t k = 1; k < temp_ciphers.size(); k++)
        c_line += temp_ciphers[k];

      fhecompiler::Ciphertext output("output" + std::to_string(i), fhecompiler::VarType::output);
      output = c_line;
    }

    fhecompiler::compile("matrix_mul.hpp");
  }
  catch (const char *message)
  {
    std::cout << message << "\n";
  }

  return 0;
}