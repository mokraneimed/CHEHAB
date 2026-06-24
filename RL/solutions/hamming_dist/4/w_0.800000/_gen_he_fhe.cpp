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
Ciphertext c46 = encrypted_inputs.at("c0");
Ciphertext c33 = encrypted_inputs.at("c7");
Ciphertext c34 = encrypted_inputs.at("c7");
Plaintext p35 = encoded_inputs.at("p5");
Plaintext p36 = encoded_inputs.at("p5");
Ciphertext c37 = encrypted_inputs.at("c8");
Ciphertext c38 = encrypted_inputs.at("c8");
Ciphertext c39 = encrypted_inputs.at("c4");
Ciphertext c40 = encrypted_inputs.at("c4");
Ciphertext c41 = encrypted_inputs.at("c3");
Ciphertext c42 = encrypted_inputs.at("c3");
Ciphertext c43 = encrypted_inputs.at("c1");
Ciphertext c44 = encrypted_inputs.at("c1");
Ciphertext c45 = encrypted_inputs.at("c0");
evaluator.add(c44, c45, c44);
evaluator.multiply(c40, c42, c40);
evaluator.relinearize(c40, relin_keys, c40);
evaluator.add_plain(c40, p36, c40);
evaluator.multiply(c34, c38, c34);
evaluator.relinearize(c34, relin_keys, c34);
evaluator.multiply(c40, c34, c40);
evaluator.relinearize(c40, relin_keys, c40);
evaluator.sub(c44, c40, c44);
evaluator.rotate_rows(c44, 2, galois_keys, c40);
evaluator.add(c44, c40, c44);
evaluator.rotate_rows(c44, 1, galois_keys, c45);
evaluator.add(c44, c45, c44);
encrypted_outputs.emplace("c32", move(c44));
}

vector<int> get_rotation_steps_fhe(){
return vector<int>{2, 1};
}

