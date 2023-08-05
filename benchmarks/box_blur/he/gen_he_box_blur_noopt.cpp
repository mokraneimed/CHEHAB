#include <cstddef>
#include <cstdint>
#include <utility>
#include "gen_he_box_blur_noopt.hpp"

using namespace std;
using namespace seal;

void box_blur_noopt(const unordered_map<string, Ciphertext> &encrypted_inputs,
const unordered_map<string, Plaintext> &encoded_inputs,
unordered_map<string, Ciphertext> &encrypted_outputs,
unordered_map<string, Plaintext> &encoded_outputs,
const BatchEncoder &encoder,
const Encryptor &encryptor,
const Evaluator &evaluator,
const RelinKeys &relin_keys,
const GaloisKeys &galois_keys)
{
Ciphertext c1 = encrypted_inputs.at("img");
size_t slot_count = encoder.slot_count();
Plaintext p24;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p24);
Plaintext p27;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p27);
Plaintext p14;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p14);
Plaintext p10;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p10);
Plaintext p20;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p20);
Plaintext p7;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p7);
Plaintext p30;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p30);
Plaintext p17;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p17);
Plaintext p4;
encoder.encode(vector<std::uint64_t>(slot_count, 1), p4);
Ciphertext c2;
evaluator.rotate_rows(c1, 992, galois_keys, c2);
Ciphertext c5;
evaluator.rotate_rows(c2, 1023, galois_keys, c5);
evaluator.multiply_plain(c5, p4, c5);
Ciphertext c37;
evaluator.multiply_plain(c2, p7, c37);
evaluator.add(c5, c37, c5);
evaluator.rotate_rows(c2, 1, galois_keys, c2);
evaluator.multiply_plain(c2, p10, c2);
evaluator.add(c5, c2, c5);
evaluator.rotate_rows(c1, 1023, galois_keys, c2);
evaluator.multiply_plain(c2, p14, c2);
evaluator.multiply_plain(c1, p17, c37);
evaluator.add(c2, c37, c2);
evaluator.rotate_rows(c1, 1, galois_keys, c37);
evaluator.multiply_plain(c37, p20, c37);
evaluator.add(c2, c37, c2);
evaluator.add(c5, c2, c5);
evaluator.rotate_rows(c1, 32, galois_keys, c1);
evaluator.rotate_rows(c1, 1023, galois_keys, c2);
evaluator.multiply_plain(c2, p24, c2);
evaluator.multiply_plain(c1, p27, c37);
evaluator.add(c2, c37, c2);
evaluator.rotate_rows(c1, 1, galois_keys, c1);
evaluator.multiply_plain(c1, p30, c1);
evaluator.add(c2, c1, c2);
evaluator.add(c5, c2, c5);
encrypted_outputs.emplace("result", move(c5));
}

vector<int> get_rotation_steps_box_blur_noopt(){
return vector<int>{992, 1023, 1, 32};
}