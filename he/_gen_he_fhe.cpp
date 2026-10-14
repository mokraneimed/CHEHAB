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
Ciphertext c343 = encrypted_inputs.at("c0");
Ciphertext c342 = encrypted_inputs.at("c0");
Ciphertext c341 = encrypted_inputs.at("c1");
Ciphertext c340 = encrypted_inputs.at("c1");
evaluator.multiply(c340, c342, c340);
evaluator.relinearize(c340, relin_keys, c340);
evaluator.rotate_rows(c340, 16, galois_keys, c341);
evaluator.add(c340, c341, c340);
evaluator.rotate_rows(c340, 8, galois_keys, c341);
evaluator.add(c340, c341, c340);
evaluator.rotate_rows(c340, 4, galois_keys, c341);
evaluator.add(c340, c341, c340);
evaluator.rotate_rows(c340, 2, galois_keys, c341);
evaluator.add(c340, c341, c340);
evaluator.rotate_rows(c340, 1, galois_keys, c341);
evaluator.add(c340, c341, c340);
encrypted_outputs.emplace("c95", move(c340));
}

vector<int> get_rotation_steps_fhe(){
return vector<int>{16, 8, 4, 2, 1};
}

