#include "fhecompiler/fhecompiler.hpp"
#include "frontend_tests_const.hpp"
#include <bits/stdc++.h>

inline void swap(fhecompiler::Ciphertext &ct1, fhecompiler::Ciphertext &ct2)
{
  fhecompiler::Ciphertext tmp = ct1;
  ct1 = ct2; //
  ct2 = tmp;
}

int test6_frontend()
{

  try
  {
    fhecompiler::init("test6", fhecompiler::Scheme::bfv, fhecompiler::Backend::SEAL);

    fhecompiler::Ciphertext ct1("ct1", fhecompiler::VarType::input);

    fhecompiler::Ciphertext ct2("ct2", fhecompiler::VarType::input);

    fhecompiler::Ciphertext ct2_otuput("ct2_output", fhecompiler::VarType::output);
    fhecompiler::Ciphertext ct1_output("ct1_output", fhecompiler::VarType::output);

    ct1 += ct1;

    ct1 += ct2;

    ct1 *= fhecompiler::exponentiate(ct1, 3);

    ct2 = ct1 - (ct1 + ct2);

    ct1_output = ct1;

    ct2_otuput = ct2;

    swap(ct1_output, ct2_otuput);

    size_t polynomial_modulus_degree = 4096;
    size_t plaintext_modulus = 786433;

    params_selector::EncryptionParameters params;
    params.set_plaintext_modulus(plaintext_modulus);
    params.set_polynomial_modulus_degree(polynomial_modulus_degree);

    fhecompiler::compile(out_dir + "/" + "test6.hpp", &params);
  }
  catch (const char *emessage)
  {
    std::cout << emessage << '\n';
  }

  return 0;
}