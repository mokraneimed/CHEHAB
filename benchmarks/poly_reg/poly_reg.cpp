#include "fheco/fheco.hpp"

using namespace std;
using namespace fheco;
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
void fhe()
{
  size_t size = 4;
  std::vector<Ciphertext> v0(size);
  std::vector<Ciphertext> v1(size);
  std::vector<Ciphertext> v2(size);
  std::vector<Ciphertext> v3(size);
  std::vector<Ciphertext> v4(size);
  std::vector<Ciphertext> output(size);
  for (int i = 0; i < size; i++)
  {
    v0[i] = Ciphertext("v0_" + std::to_string(i));
  }
  for (int i = 0; i < size; i++)
  {
    v1[i] = Ciphertext("v1_" + std::to_string(i));
  }
  for (int i = 0; i < size; i++)
  {
    v2[i] = Ciphertext("v2_" + std::to_string(i));
  }
  for (int i = 0; i < size; i++)
  {
    v3[i] = Ciphertext("v3_" + std::to_string(i));
  }

  for (int i = 0; i < size; i++)
  {

    v4[i] = Ciphertext("v4_" + std::to_string(i));
    output[i] = v1[i] - (v4[i] * v0[i] * v0[i] + v3[i] * v0[i] + v2[i]);
  }

  for (int i = 0; i < size; i++)
  {
    output[i].set_output("output_" + std::to_string(i));
  }
}
void print_bool_arg(bool arg, const string &name, ostream &os)
{
  os << (arg ? name : "no_" + name);
}
int main(int argc, char **argv)
{
  bool vectorized = true;
  if (argc > 1)
    vectorized = stoi(argv[1]);

  int window = 1;
  if (argc > 2)
    window = stoi(argv[2]);

  bool call_quantifier = false;
  if (argc > 3)
    call_quantifier = stoi(argv[3]);

  bool cse = true;
  if (argc > 4)
    cse = stoi(argv[4]);

  bool const_folding = true;
  if (argc > 5)
    const_folding = stoi(argv[5]);

  if (cse)
  {
    Compiler::enable_cse();
    Compiler::enable_order_operands();
  }
  else
  {
    Compiler::disable_cse();
    Compiler::disable_order_operands();
  }

  if (const_folding)
    Compiler::enable_const_folding();
  else
    Compiler::disable_const_folding();

  chrono::high_resolution_clock::time_point t;
  chrono::duration<double, milli> elapsed;
  t = chrono::high_resolution_clock::now();
  string func_name = "fhe";
  const auto &func = Compiler::create_func(func_name, 1, 20, false, true);
  fhe();

  if (vectorized)
  {
    cout << " window is " << window << endl;
    Compiler::gen_vectorized_code(func, window);
  }
  else
  {
    string gen_name = "_gen_he_" + func_name;
    string gen_path = "he/" + gen_name;
    ofstream header_os(gen_path + ".hpp");
    if (!header_os)
      throw logic_error("failed to create header file");

    ofstream source_os(gen_path + ".cpp");
    if (!source_os)
      throw logic_error("failed to create source file");

    if (call_quantifier)
    {
      util::Quantifier quantifier{func};
      quantifier.run_all_analysis();
      quantifier.print_info(cout);
    }
  }

  elapsed = chrono::high_resolution_clock::now() - t;
  cout << elapsed.count() << " ms\n";

  return 0;
}