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
Ciphertext c38 = encrypted_inputs.at("c0");
Ciphertext c37 = encrypted_inputs.at("c0");
Ciphertext c36 = encrypted_inputs.at("c3");
Ciphertext c35 = encrypted_inputs.at("c3");
Ciphertext c34 = encrypted_inputs.at("c4");
Ciphertext c33 = encrypted_inputs.at("c4");
evaluator.rotate_rows(c37, 4, galois_keys, c34);
evaluator.add(c37, c34, c37);
evaluator.multiply(c33, c35, c33);
evaluator.relinearize(c33, relin_keys, c33);
evaluator.rotate_rows(c33, 4, galois_keys, c34);
evaluator.multiply(c33, c34, c33);
evaluator.relinearize(c33, relin_keys, c33);
evaluator.sub(c37, c33, c37);
evaluator.rotate_rows(c37, 2, galois_keys, c34);
evaluator.add(c37, c34, c37);
evaluator.rotate_rows(c37, 1, galois_keys, c34);
evaluator.add(c37, c34, c37);
encrypted_outputs.emplace("c36", move(c37));
}

vector<int> get_rotation_steps_fhe(){
return vector<int>{4, 2, 1};
}

