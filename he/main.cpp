#include <chrono>
    #include <cstddef>
    #include <fstream>
    #include <iostream>
    #include <ostream> 
    #include "_gen_he_fhe.hpp"
    #include "utils.hpp"

    using namespace std;
    using namespace seal;

    void print_bool_arg(bool arg, const string &name, ostream &os)
    {
      os << (arg ? name : "no" + name); 
    }

    int main(int argc, char **argv)
    {
      bool opt = true;
      if (argc > 1)
        opt = stoi(argv[1]);

      print_bool_arg(opt, "opt", clog);
      clog << '\n';

      string app_name = "fhe";
      ifstream is("../" + app_name + "_io_example_adapted.txt");
      if (!is)
        throw invalid_argument("failed to open io example file");
        EncryptionParameters params(scheme_type::bfv); 
  size_t n =16384 ;
  params.set_poly_modulus_degree(n);
  params.set_plain_modulus(PlainModulus::Batching(n, 20)); 
params.set_coeff_modulus(CoeffModulus::BFVDefault(16384));
  SEALContext context(params, true, sec_level_type::tc128);

      ClearArgsInfo clear_inputs, clear_outputs;
      size_t func_slot_count;
      BatchEncoder batch_encoder(context);
      KeyGenerator keygen(context);
      const SecretKey &secret_key = keygen.secret_key();
      PublicKey public_key;
      keygen.create_public_key(public_key);
      RelinKeys relin_keys;
      keygen.create_relin_keys(relin_keys);
      GaloisKeys galois_keys; 
      chrono::high_resolution_clock::time_point keys_time;
      chrono::duration<double, milli> keys_elapsed;
      keys_time = chrono::high_resolution_clock::now();      
      //keygen.create_galois_keys(galois_keys);
      keygen.create_galois_keys(get_rotation_steps_fhe(), galois_keys);
      keys_elapsed = chrono::high_resolution_clock::now() - keys_time;

      size_t galois_keys_total_size = galois_keys.save_size();
      cout << "rotation_keys_size_(MB): " << galois_keys_total_size / (1024.0 * 1024.0) << endl;
      Encryptor encryptor(context, public_key);
      Evaluator evaluator(context);
      Decryptor decryptor(context, secret_key);
      EncryptedArgs encrypted_inputs;
      EncodedArgs encoded_inputs;
      
      chrono::high_resolution_clock::time_point t;
      chrono::duration<double, milli> elapsed;
      t = chrono::high_resolution_clock::now();

      parse_inputs_outputs_file(is, params.plain_modulus().value(), clear_inputs, clear_outputs, func_slot_count);
      prepare_he_inputs(batch_encoder, encryptor, clear_inputs, encrypted_inputs, encoded_inputs);
      EncryptedArgs encrypted_outputs;
      EncodedArgs encoded_outputs;

      fhe(
        encrypted_inputs, encoded_inputs, encrypted_outputs, encoded_outputs, batch_encoder, encryptor, evaluator,
        relin_keys, galois_keys);

      elapsed = chrono::high_resolution_clock::now() - t;

      ClearArgsInfo obtained_clear_outputs;
      get_clear_outputs(
        batch_encoder, decryptor, encrypted_outputs, encoded_outputs, func_slot_count, obtained_clear_outputs);
      print_encrypted_outputs_info(context, decryptor, encrypted_outputs, clog);
      cout <<"circuit_execution_time_(ms): "<<elapsed.count() <<"\n"<<std::flush;
      cout <<"galois_keys_generation_time_(ms): "<<keys_elapsed.count() <<"\n"<<std::flush;
      double total_time = elapsed.count() + keys_elapsed.count();
      cout <<"total_execution_time_(ms): "<<total_time <<"\n"<<std::flush;
      print_variables_values(obtained_clear_outputs, std::cout);
    }
    