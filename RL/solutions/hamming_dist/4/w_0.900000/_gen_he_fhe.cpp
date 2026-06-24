#include <cstddef>
#include <cstdint>
#include <utility>
#include "_gen_he_fhe.hpp"

using namespace std;
using namespace seal;

void fhe(const unordered_map<string, Ciphertext> &encrypted_inputs,
const unordered_map<string, Plaintext> &encoded_inputs,
unordered_map<string, Ciphertext> &encrypted_outputs,
unordered_map<string, Plaintext> &encoded_outputs,
const BatchEncoder &encoder,
const Encryptor &encryptor,
const Evaluator &evaluator,
const RelinKeys &relin_keys,
const GaloisKeys &galois_keys)
{
Ciphertext c44 = encrypted_inputs.at("c0");
Ciphertext c43 = encrypted_inputs.at("c0");
Ciphertext c42 = encrypted_inputs.at("c3");
Ciphertext c41 = encrypted_inputs.at("c3");
Ciphertext c40 = encrypted_inputs.at("c4");
Ciphertext c39 = encrypted_inputs.at("c4");
Ciphertext c38 = encrypted_inputs.at("c8");
Ciphertext c37 = encrypted_inputs.at("c8");
Plaintext p36 = encoded_inputs.at("p5");
Plaintext p35 = encoded_inputs.at("p5");
Ciphertext c34 = encrypted_inputs.at("c7");
Ciphertext c33 = encrypted_inputs.at("c7");
evaluator.rotate_rows(c43, 4, galois_keys, c34);
evaluator.add(c43, c34, c43);
evaluator.multiply(c39, c41, c39);
evaluator.relinearize(c39, relin_keys, c39);
evaluator.add_plain(c39, p35, c39);
evaluator.multiply(c33, c37, c33);
evaluator.relinearize(c33, relin_keys, c33);
evaluator.multiply(c39, c33, c39);
evaluator.relinearize(c39, relin_keys, c39);
evaluator.sub(c43, c39, c43);
evaluator.rotate_rows(c43, 2, galois_keys, c33);
evaluator.add(c43, c33, c43);
evaluator.rotate_rows(c43, 1, galois_keys, c39);
evaluator.add(c43, c39, c43);
encrypted_outputs.emplace("c35", move(c43));
}

vector<int> get_rotation_steps_fhe(){
return vector<int>{4, 2, 1};
}

