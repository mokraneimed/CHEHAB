#include "fheco/code_gen/gen_func_lattigo.hpp"
#include "fheco/code_gen/constants_lattigo.hpp"
#include "fheco/ckks/ckks_params.hpp"
#include "fheco/ir/common.hpp"
#include "fheco/ir/func.hpp"
#include "fheco/passes/prepare_code_gen.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

namespace fheco::code_gen::lattigo
{

namespace
{

string reference_name(size_t term_id)
{
  return "ref_" + to_string(term_id);
}

bool can_generate_primitive_reference(const shared_ptr<ir::Func> &func)
{
  unordered_set<size_t> supported;
  for (const auto &input : func->data_flow().inputs_info())
    if (input.first->type() == ir::Term::Type::cipher)
      supported.insert(input.first->id());
  for (const auto &constant : func->data_flow().constants_info())
    supported.insert(constant.first->id());

  for (const auto *term : func->get_top_sorted_terms())
  {
    if (!term->is_operation())
      continue;

    const auto type = term->op_code().type();
    const auto required_operands = type == ir::OpCode::Type::add ||
      type == ir::OpCode::Type::sub || type == ir::OpCode::Type::mul ? 2 : 1;
    if (term->operands().size() < required_operands)
      return false;
    for (size_t index = 0; index < required_operands; ++index)
      if (!supported.count(term->operands()[index]->id()))
        return false;

    switch (type)
    {
      case ir::OpCode::Type::encrypt:
      case ir::OpCode::Type::add:
      case ir::OpCode::Type::sub:
      case ir::OpCode::Type::negate:
      case ir::OpCode::Type::rotate:
      case ir::OpCode::Type::square:
      case ir::OpCode::Type::mul:
      case ir::OpCode::Type::mod_switch:
      case ir::OpCode::Type::relin:
      case ir::OpCode::Type::SumVec:
      case ir::OpCode::Type::rescale:
      case ir::OpCode::Type::bootstrap:
        supported.insert(term->id());
        break;
      default:
        return false;
    }
  }

  for (const auto &output : func->data_flow().outputs_info())
    if (output.first->type() != ir::Term::Type::cipher || !supported.count(output.first->id()))
      return false;
  return true;
}

void gen_primitive_reference_go(const shared_ptr<ir::Func> &func, ostream &os)
{
  unordered_map<size_t, string> references;
  for (const auto &input : func->data_flow().inputs_info())
  {
    if (input.first->type() != ir::Term::Type::cipher)
      continue;
    const auto id = input.first->id();
    const auto ref = reference_name(id);
    references.emplace(id, ref);
    os << "\t" << ref << " := append([]float64(nil), values...)\n";
  }

  for (const auto &constant : func->data_flow().constants_info())
  {
    const auto id = constant.first->id();
    const auto ref = reference_name(id);
    references.emplace(id, ref);
    os << "\t" << ref << " := make([]float64, len(values))\n";
    if (constant.second.is_scalar_)
    {
      os << "\tfor i := range " << ref << " { " << ref << "[i] = "
         << constant.second.val_[0] << " }\n";
    }
    else
    {
      const auto &vec = constant.second.val_;
      size_t nz = 0;
      for (double v : vec)
      {
        if (v != 0.0) ++nz;
      }
      if (nz * 4 < vec.size())
      {
        for (size_t index = 0; index < vec.size(); ++index)
        {
          if (vec[index] != 0.0)
            os << "\t" << ref << "[" << index << "] = " << vec[index] << "\n";
        }
      }
      else
      {
        os << "\tcopy(" << ref << ", []float64{";
        for (size_t index = 0; index < constant.second.val_.size(); ++index)
        {
          if (index) os << ", ";
          os << constant.second.val_[index];
        }
        os << "})\n";
      }
    }
  }

  for (const auto *term : func->get_top_sorted_terms())
  {
    if (!term->is_operation())
      continue;

    const auto ref = reference_name(term->id());
    const auto operand = references.at(term->operands()[0]->id());
    references.emplace(term->id(), ref);
    const auto type = term->op_code().type();
    if (type == ir::OpCode::Type::encrypt || type == ir::OpCode::Type::mod_switch ||
        type == ir::OpCode::Type::relin || type == ir::OpCode::Type::rescale ||
        type == ir::OpCode::Type::bootstrap)
    {
      os << "\t" << ref << " := " << operand << "\n";
    }
    else if (type == ir::OpCode::Type::rotate)
    {
      os << "\t" << ref << " := make([]float64, len(" << operand << "))\n";
      os << "\tfor i := range " << ref << " { " << ref << "[i] = " << operand
         << "[(i + " << term->op_code().steps() << " + len(" << operand << ")) % len(" << operand << ")] }\n";
    }
    else if (type == ir::OpCode::Type::SumVec)
    {
      os << "\t" << ref << " := append([]float64(nil), " << operand << "...)\n";
      for (int step = term->op_code().size() / 2; step >= 1; step /= 2)
      {
        os << "\t{ next := make([]float64, len(" << ref << ")); for i := range next { next[i] = "
           << ref << "[i] + " << ref << "[(i + " << step << ") % len(" << ref << ")] }; "
           << ref << " = next }\n";
      }
    }
    else
    {
      os << "\t" << ref << " := make([]float64, len(" << operand << "))\n";
      if (type == ir::OpCode::Type::negate)
      {
        os << "\tfor i := range " << ref << " { " << ref << "[i] = -" << operand << "[i] }\n";
      }
      else if (type == ir::OpCode::Type::square)
      {
        os << "\tfor i := range " << ref << " { " << ref << "[i] = " << operand << "[i] * " << operand << "[i] }\n";
      }
      else
      {
        const auto operand2 = references.at(term->operands()[1]->id());
        const char *op = type == ir::OpCode::Type::add ? "+" :
          type == ir::OpCode::Type::sub ? "-" : "*";
        os << "\tfor i := range " << ref << " { " << ref << "[i] = " << operand
           << "[i] " << op << " " << operand2 << "[i] }\n";
      }
    }
  }

  os << "\texpectedOutputs := make(map[string][]float64)\n";
  for (const auto &output : func->data_flow().outputs_info())
  {
    const auto &ref = references.at(output.first->id());
    for (const auto &label : output.second.labels_)
      os << "\texpectedOutputs[\"" << label << "\"] = " << ref << "\n";
  }
}

} // namespace

void gen_func_lattigo(
  const shared_ptr<ir::Func> &func,
  const unordered_set<int> &rotation_steps,
  ostream &os,
  const string &func_name,
  const ckks::CKKSParams* ckks_params)
{
  passes::prepare_code_gen(func);
  
  // Write Go file header with imports (include bootstrap imports if needed)
  if (ckks_params && ckks_params->enable_bootstrap)
  {
    os << go_file_header_bootstrap;
  }
  else if (ckks_params && (ckks_params->ring_type == ckks::RingType::ConjugateInvariant || ckks_params->hamming_weight == 8192))
  {
    os << go_file_header_ring;
  }
  else
  {
    os << go_file_header;
  }
  
  // Generate rotation steps getter
  gen_rotation_steps_getter_go(func_name, rotation_steps, os);
  os << "\n";
  
  // Generate the main computation function
  gen_func_signature_go(func_name, os);
  os << " {\n";
  
  TermsCtxtObjectsInfo terms_ctxt_objects_info;
  gen_input_terms_go(func->data_flow().inputs_info(), os, terms_ctxt_objects_info);
  gen_const_terms_go(func->data_flow().constants_info(), func->clear_data_evaluator().signedness(), os);
  gen_op_terms_go(func, os, terms_ctxt_objects_info);
  gen_output_terms_go(func->data_flow().outputs_info(), os, terms_ctxt_objects_info);
  
  os << "}\n\n";
  
  // Collect unique cipher input labels for main() input preparation
  std::set<std::string> cipher_input_labels;
  for (const auto &input_info : func->data_flow().inputs_info())
  {
    if (input_info.first->type() == ir::Term::Type::cipher)
      cipher_input_labels.insert(input_info.second.label_);
  }

  std::size_t bootstrap_count = 0;
  for (auto term : func->get_top_sorted_terms())
    bootstrap_count += term->is_operation() && term->op_code().type() == ir::OpCode::Type::bootstrap;

  // Generate main function with setup (pass CKKS params)
  gen_main_go(func_name, rotation_steps, os, func, ckks_params, cipher_input_labels, bootstrap_count);
}

void gen_func_signature_go(const string &func_name, ostream &os)
{
  os << "func " << func_name << "(\n";
  os << "\tencryptedInputs map[string]*rlwe.Ciphertext,\n";
  os << "\tencodedInputs map[string]*rlwe.Plaintext,\n";
  os << "\tencryptedOutputs map[string]*rlwe.Ciphertext,\n";
  os << "\tencodedOutputs map[string]*rlwe.Plaintext,\n";
  os << "\tencoder *hefloat.Encoder,\n";
  os << "\tenc *rlwe.Encryptor,\n";
  os << "\teval *hefloat.Evaluator,\n";
  os << "\tparams hefloat.Parameters,\n";
  os << ")";
}

void gen_cipher_var_id_go(size_t term_id, ostream &os)
{
  os << "c" << term_id;
}

void gen_plain_var_id_go(size_t term_id, ostream &os)
{
  os << "p" << term_id;
}

void gen_input_terms_go(
  const ir::InputTermsInfo &input_terms_info,
  ostream &os,
  TermsCtxtObjectsInfo &terms_ctxt_objects_info)
{
  for (const auto &input_info : input_terms_info)
  {
    auto term = input_info.first;
    auto object_id = term->id();
    
    if (term->type() == ir::Term::Type::cipher)
    {
      terms_ctxt_objects_info.emplace(term->id(), CtxtObjectInfo{object_id, term->parents().size()});
      os << "\t";
      gen_cipher_var_id_go(object_id, os);
      os << " := encryptedInputs[\"" << input_info.second.label_ << "\"]\n";
      os << "\t_ = ";
      gen_cipher_var_id_go(object_id, os);
      os << "\n";
    }
    else
    {
      os << "\t";
      gen_plain_var_id_go(object_id, os);
      os << " := encodedInputs[\"" << input_info.second.label_ << "\"]\n";
      os << "\t_ = ";
      gen_plain_var_id_go(object_id, os);
      os << "\n";
    }
  }
}

void gen_const_terms_go(
  const ir::ConstTermsValues &const_terms_info,
  bool signedness,
  ostream &os)
{
  if (const_terms_info.empty())
    return;
    
  os << "\n\t// Encode constants\n";
  os << "\tslotCount := params.MaxSlots()\n";
  os << "\t_ = slotCount\n";
  
  for (const auto &const_info : const_terms_info)
  {
    auto term = const_info.first;
    auto object_id = term->id();
    
    os << "\t";
    gen_plain_var_id_go(object_id, os);
    os << " := hefloat.NewPlaintext(params, params.MaxLevel())\n";
    
    if (const_info.second.is_scalar_)
    {
      // Scalar constant - replicate across all slots
      os << "\t{\n";
      os << "\t\tvalues := make([]float64, slotCount)\n";
      os << "\t\tfor i := range values {\n";
      os << "\t\t\tvalues[i] = float64(" << const_info.second.val_[0] << ")\n";
      os << "\t\t}\n";
      os << "\t\tencoder.Encode(values, ";
      gen_plain_var_id_go(object_id, os);
      os << ")\n";
      os << "\t}\n";
    }
    else
    {
      // Vector constant
      const auto &vec = const_info.second.val_;
      size_t nz = 0;
      for (double v : vec)
      {
        if (v != 0.0) ++nz;
      }
      os << "\t{\n";
      if (nz * 4 < vec.size())
      {
        os << "\t\tvalues := make([]float64, slotCount)\n";
        for (size_t i = 0; i < vec.size(); ++i)
        {
          if (vec[i] != 0.0)
          {
            os << "\t\tvalues[" << i << "] = float64(" << vec[i] << ")\n";
          }
        }
      }
      else
      {
        os << "\t\tvalues := []float64{";
        for (size_t i = 0; i < const_info.second.val_.size(); ++i)
        {
          if (i > 0) os << ", ";
          os << "float64(" << const_info.second.val_[i] << ")";
        }
        os << "}\n";
      }
      os << "\t\tencoder.Encode(values, ";
      gen_plain_var_id_go(object_id, os);
      os << ")\n";
      os << "\t}\n";
    }
  }
}

void gen_op_terms_go(
  const shared_ptr<ir::Func> &func,
  ostream &os,
  TermsCtxtObjectsInfo &terms_ctxt_objects_info)
{
  os << "\n\t// FHE Operations\n";

  std::unordered_map<size_t, std::vector<const ir::Term *>> rotation_clusters;
  for (auto term : func->get_top_sorted_terms())
    if (term->is_operation() && term->op_code().type() == ir::OpCode::Type::rotate)
      rotation_clusters[term->operands()[0]->id()].push_back(term);

  std::unordered_set<size_t> hoisted_inputs;
  for (const auto &[input_id, rotations] : rotation_clusters)
    if (rotations.size() >= 2)
      hoisted_inputs.insert(input_id);
  if (!hoisted_inputs.empty())
  {
    os << "\tbuffDecompQP := eval.GetBuffDecompQP()\n";
    os << "\tvar lastDecompID int = -1\n";
  }
  
  for (auto term : func->get_top_sorted_terms())
  {
    if (!term->is_operation())
      continue;

    auto term_object_id = term->id();
    vector<size_t> operands_ctxt_objects_ids(term->operands().size());
    unordered_map<size_t, size_t> operands_multip;
    
    for (size_t i = 0; i < operands_ctxt_objects_ids.size(); ++i)
    {
      auto operand = term->operands()[i];
      if (operand->type() != ir::Term::Type::cipher)
        continue;

      auto multip = ++operands_multip[operand->id()];
      auto operand_object_info_it = terms_ctxt_objects_info.find(operand->id());
      
      if (operand_object_info_it == terms_ctxt_objects_info.end())
      {
        operands_ctxt_objects_ids[i] = term_object_id;
        continue;
      }
      
      auto &operand_object_info = operand_object_info_it->second;
      operands_ctxt_objects_ids[i] = operand_object_info.id_;
      
      if (func->data_flow().is_output(operand) || operand->type() == ir::Term::Type::plain || multip > 1)
        continue;

      --operand_object_info.dep_count_;
      if (term_object_id == term->id() && operand_object_info.dep_count_ == 0)
      {
        term_object_id = operands_ctxt_objects_ids[i];
        terms_ctxt_objects_info.erase(operand_object_info_it);
      }
    }

    if (term_object_id == term->id())
    {
      for (auto it = terms_ctxt_objects_info.begin(); it != terms_ctxt_objects_info.end(); ++it)
      {
        if (it->second.dep_count_ == 0)
        {
          term_object_id = it->second.id_;
          terms_ctxt_objects_info.erase(it);
          break;
        }
      }
    }
    
    auto dep_count = term->parents().size();
    if (func->data_flow().is_output(term))
      ++dep_count;

    terms_ctxt_objects_info.emplace(term->id(), CtxtObjectInfo{term_object_id, dep_count});

    // Declare new ciphertext if needed
    if (term_object_id == term->id())
    {
      os << "\tvar ";
      gen_cipher_var_id_go(term_object_id, os);
      os << " *rlwe.Ciphertext\n";
    }

    // Generate the operation
    vector<ir::Term::Type> operands_types;
    operands_types.reserve(term->operands().size());
    transform(
      term->operands().cbegin(), term->operands().cend(), back_inserter(operands_types),
      [](const ir::Term *operand) { return operand->type(); });

    if (term->op_code() == ir::OpCode::encrypt)
    {
      // Encrypt operation
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = enc.EncryptNew(";
      gen_plain_var_id_go(term->operands()[0]->id(), os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::rotate)
    {
      int steps = term->op_code().steps();
      const auto input_id = term->operands()[0]->id();
      const auto input_object_id = operands_ctxt_objects_ids[0];
      if (hoisted_inputs.count(input_id) && steps != 0)
      {
        os << "\tif lastDecompID != " << input_object_id << " {\n";
        os << "\t\teval.DecomposeNTT(";
        gen_cipher_var_id_go(input_object_id, os);
        os << ".Level(), params.MaxLevelP(), params.PCount(), ";
        gen_cipher_var_id_go(input_object_id, os);
        os << ".Value[1], ";
        gen_cipher_var_id_go(input_object_id, os);
        os << ".IsNTT, buffDecompQP)\n";
        os << "\t\tlastDecompID = " << input_object_id << "\n\t}\n";
        os << "\tif ";
        gen_cipher_var_id_go(term_object_id, os);
        os << " == nil { ";
        gen_cipher_var_id_go(term_object_id, os);
        os << " = rlwe.NewCiphertext(params, 1, ";
        gen_cipher_var_id_go(input_object_id, os);
        os << ".Level()) }\n";
        os << "\t_ = eval.AutomorphismHoisted(";
        gen_cipher_var_id_go(input_object_id, os);
        os << ".Level(), ";
        gen_cipher_var_id_go(input_object_id, os);
        os << ", buffDecompQP, params.GaloisElement(" << steps << "), ";
        gen_cipher_var_id_go(term_object_id, os);
        os << ")\n";
      }
      else
      {
        os << "\t";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", _ = eval.RotateNew(";
        gen_cipher_var_id_go(input_object_id, os);
        os << ", " << steps << ")\n";
      }
    }
    else if (term->op_code().type() == ir::OpCode::Type::square)
    {
      // Square: eval.MulRelinNew(ct, ct) + auto-rescale for CKKS
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = eval.MulRelinNew(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ", ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ")\n";
      // Auto-rescale after square (cipher-cipher multiplication)
      os << "\t_ = eval.Rescale(";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", ";
      gen_cipher_var_id_go(term_object_id, os);
      os << ")\n";
      if (!hoisted_inputs.empty())
        os << "\tlastDecompID = -1\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::rescale)
    {
      // Rescale: copy input, then rescale in-place
      // Lattigo's Rescale modifies in-place, so we need to copy first
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ".CopyNew()\n";
      os << "\t_ = eval.Rescale(";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", ";
      gen_cipher_var_id_go(term_object_id, os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::relin)
    {
      // Relinearize: copy input, then relinearize in-place
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ".CopyNew()\n";
      os << "\t_ = eval.Relinearize(";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", ";
      gen_cipher_var_id_go(term_object_id, os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::mod_switch)
    {
      // DropLevel
      os << "\teval.DropLevel(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ", 1)\n";
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << "\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::SumVec)
    {
      // SumVec reduction: sum all slots using log(n) rotations and additions
      // SumVec(x, size) → x + (x << size/2) + ((x + (x << size/2)) << size/4) + ...
      int size = term->op_code().size();
      os << "\t// SumVec reduction (size=" << size << ")\n";
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << " = ";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ".CopyNew()\n";
      
      // Generate log2(size) rotations and additions
      int step = size / 2;
      while (step >= 1)
      {
        os << "\t{\n";
        os << "\t\trotated, _ := eval.RotateNew(";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", " << step << ")\n";
        os << "\t\t";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", _ = eval.AddNew(";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", rotated)\n";
        os << "\t}\n";
        step /= 2;
      }
    }
    else if (term->op_code().type() == ir::OpCode::Type::bootstrap)
    {
      // Bootstrap: refresh ciphertext to max level
      // Requires bootstrapper to be initialized (see gen_main_go)
      os << "\t// Bootstrap: refresh to max level\n";
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = bootstrapper.Bootstrap(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ")\n";
    }
    else if (term->op_code().type() == ir::OpCode::Type::negate)
    {
      // Negate: eval.NegNew(ct)
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = eval.NegNew(";
      gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      os << ")\n";
    }
    else
    {
      // Binary operations: add, sub, mul
      auto op_type = ir::OpType{term->op_code().type(), std::move(operands_types)};
      auto op_it = operation_mapping.find(op_type);
      
      if (op_it == operation_mapping.end())
      {
        os << "\t// WARNING: Unsupported operation " << term->op_code() << "\n";
        continue;
      }
      
      string op_name(op_it->second);
      
      os << "\t";
      gen_cipher_var_id_go(term_object_id, os);
      os << ", _ = eval." << op_name << "New(";
      
      // First operand (always cipher for these ops)
      auto operand0 = term->operands()[0];
      if (operand0->type() == ir::Term::Type::cipher)
        gen_cipher_var_id_go(operands_ctxt_objects_ids[0], os);
      else
        gen_plain_var_id_go(operand0->id(), os);
      
      os << ", ";
      
      // Second operand
      auto operand1 = term->operands()[1];
      if (operand1->type() == ir::Term::Type::cipher)
        gen_cipher_var_id_go(operands_ctxt_objects_ids[1], os);
      else
        gen_plain_var_id_go(operand1->id(), os);
      
      os << ")\n";
      
      // Every CKKS multiplication raises the scale. A plaintext operand does
      // not remove that requirement, so normalize both multiplication forms.
      if (op_name == "MulRelin" || op_name == "Mul")
      {
        os << "\t_ = eval.Rescale(";
        gen_cipher_var_id_go(term_object_id, os);
        os << ", ";
        gen_cipher_var_id_go(term_object_id, os);
        os << ")\n";
        if (!hoisted_inputs.empty())
          os << "\tlastDecompID = -1\n";
      }
    }
  }
}

void gen_output_terms_go(
  const ir::OutputTermsInfo &output_terms_info,
  ostream &os,
  const TermsCtxtObjectsInfo &terms_ctxt_objects_info)
{
  os << "\n\t// Store outputs\n";
  
  for (const auto &output_info : output_terms_info)
  {
    auto term = output_info.first;
    
    if (term->type() == ir::Term::Type::cipher)
    {
      auto ctxt_object_id = terms_ctxt_objects_info.at(term->id()).id_;
      
      for (const auto &label : output_info.second.labels_)
      {
        os << "\tencryptedOutputs[\"" << label << "\"] = ";
        gen_cipher_var_id_go(ctxt_object_id, os);
        os << "\n";
      }
    }
    else
    {
      for (const auto &label : output_info.second.labels_)
      {
        os << "\tencodedOutputs[\"" << label << "\"] = ";
        gen_plain_var_id_go(term->id(), os);
        os << "\n";
      }
    }
  }
}

void gen_rotation_steps_getter_go(
  const string &func_name,
  const unordered_set<int> &steps,
  ostream &os)
{
  os << "func getRotationSteps() []int {\n";
  os << "\treturn []int{";
  
  bool first = true;
  for (int step : steps)
  {
    if (!first) os << ", ";
    os << step;
    first = false;
  }
  
  os << "}\n";
  os << "}\n";
}

void gen_main_go(
  const string &func_name,
  const unordered_set<int> &rotation_steps,
  ostream &os,
  const shared_ptr<ir::Func> &func,
  const ckks::CKKSParams* ckks_params,
  const set<string> &cipher_input_labels,
  size_t bootstrap_count)
{
  // Use provided params or create defaults
  ckks::CKKSParams params;
  if (ckks_params) {
    params = *ckks_params;
  } else {
    // Default params for depth ~7
    params = ckks::CKKSParamSelector::default_params(7);
  }
  
  os << "\nfunc main() {\n";
  os << "\t// CKKS Parameters (generated from CKKSParamSelector)\n";
  os << "\t// LogN=" << params.log_n << " (n=" << params.poly_modulus_degree() << ", slots=" << params.slot_count() << ")\n";
  os << "\t// MaxLevel=" << params.max_level() << ", LogScale=" << params.log_scale << "\n";
  os << "\tparams, err := hefloat.NewParametersFromLiteral(hefloat.ParametersLiteral{\n";
  os << "\t\tLogN:            " << params.log_n << ",\n";
  
  // Generate LogQ array
  os << "\t\tLogQ:            []int{";
  for (size_t i = 0; i < params.log_q.size(); ++i) {
    if (i > 0) os << ", ";
    os << params.log_q[i];
  }
  os << "},\n";
  
  // Generate LogP array
  os << "\t\tLogP:            []int{";
  for (size_t i = 0; i < params.log_p.size(); ++i) {
    if (i > 0) os << ", ";
    os << params.log_p[i];
  }
  os << "},\n";
  
  os << "\t\tLogDefaultScale: " << params.log_scale << ",\n";
  
  if (params.ring_type == ckks::RingType::ConjugateInvariant)
  {
    os << "\t\tRingType:        ring.ConjugateInvariant,\n";
  }
  else if (params.enable_bootstrap)
  {
    os << "\t\tRingType:        ring.Standard,\n";
  }

  if (params.hamming_weight > 0 && (params.enable_bootstrap || params.hamming_weight == 8192))
  {
    os << "\t\tXs:              ring.Ternary{H: " << params.hamming_weight << "},\n";
  }
  
  os << "\t})\n";
  os << "\tif err != nil {\n";
  os << "\t\tpanic(err)\n";
  os << "\t}\n\n";
  
  os << R"(	// Key Generation
	kgen := rlwe.NewKeyGenerator(params)
	sk := kgen.GenSecretKeyNew()
	pk := kgen.GenPublicKeyNew(sk)
	rlk := kgen.GenRelinearizationKeyNew(sk)

	// Galois keys for rotations
	rotations := getRotationSteps()
	galoisElements := make([]uint64, len(rotations))
	for i, r := range rotations {
		galoisElements[i] = params.GaloisElement(r)
	}

	gks := kgen.GenGaloisKeysNew(galoisElements, sk)

	var galois_keys_total_size int
	for _, gk := range gks {
		if b, err := gk.MarshalBinary(); err == nil {
			galois_keys_total_size += len(b)
		}
	}
	fmt.Printf("rotation_keys_size_(MB): %f\n", float64(galois_keys_total_size)/(1024.0*1024.0))

	evk := rlwe.NewMemEvaluationKeySet(rlk, gks...)

	// Encoder, Encryptor, Decryptor, Evaluator
	encoder := hefloat.NewEncoder(params)
	enc := rlwe.NewEncryptor(params, pk)
	dec := rlwe.NewDecryptor(params, sk)
	eval := hefloat.NewEvaluator(params, evk)
)";

  // Add bootstrapper if enabled (Orion-compatible configuration)
  if (params.enable_bootstrap)
  {
    os << "\n\t// Bootstrapper setup (Orion-compatible full configuration)\n";
    os << "\t// This configuration matches Orion's bootstrapping parameters\n";
    os << "\tbtpParamsLit := bootstrapping.ParametersLiteral{\n";
    os << "\t\tLogN: utils.Pointy(params.LogN()),\n";
    
    // LogP for bootstrapping
    os << "\t\tLogP: []int{";
    for (size_t i = 0; i < params.log_p_boot.size(); ++i) {
      if (i > 0) os << ", ";
      os << params.log_p_boot[i];
    }
    os << "},\n";
    
    // Secret key distribution (Hamming weight) - CRITICAL for bootstrap to work
    os << "\t\tXs: ring.Ternary{H: " << params.hamming_weight << "},\n";
    
    // LogSlots - number of slots to bootstrap
    int log_slots = params.effective_log_slots();
    os << "\t\tLogSlots: utils.Pointy(" << log_slots << "),\n";
    
    os << "\t}\n";
    
    os << "\tbtpParams, err := bootstrapping.NewParametersFromLiteral(params, btpParamsLit)\n";
    os << "\tif err != nil {\n";
    os << "\t\tpanic(fmt.Errorf(\"bootstrap params error: %v\", err))\n";
    os << "\t}\n";
    
    os << "\n\t// Generate bootstrap evaluation keys\n";
    os << "\tfmt.Println(\"Generating bootstrap keys (this may take a moment)...\")\n";
    os << "\tbtpKeys, _, err := btpParams.GenEvaluationKeys(sk)\n";
    os << "\tif err != nil {\n";
    os << "\t\tpanic(fmt.Errorf(\"bootstrap keygen error: %v\", err))\n";
    os << "\t}\n";
    
    os << "\n\t// Create bootstrapper evaluator (global variable)\n";
    os << "\tbootstrapper, err = bootstrapping.NewEvaluator(btpParams, btpKeys)\n";
    os << "\tif err != nil {\n";
    os << "\t\tpanic(fmt.Errorf(\"bootstrap evaluator error: %v\", err))\n";
    os << "\t}\n";
    os << "\tfmt.Println(\"Bootstrap keys generated successfully!\")\n";
  }
  
  os << R"(
	// Deterministic inputs and timing setup.
	repetitions := flag.Int("repetitions", 1, "number of encrypted inference repetitions")
	flag.Parse()
	if *repetitions < 1 { panic("repetitions must be positive") }
	encryptedInputs := make(map[string]*rlwe.Ciphertext)
	encodedInputs := make(map[string]*rlwe.Plaintext)
	encryptedOutputs := make(map[string]*rlwe.Ciphertext)
	encodedOutputs := make(map[string]*rlwe.Plaintext)
	encryptStart := time.Now()
)";

  // Generate encrypted inputs for each unique cipher input label
  os << "\tvalues := make([]float64, params.MaxSlots())\n";
  os << "\tfor i := range values { values[i] = float64((i % 17) + 1) / 17.0 }\n";
  for (const auto &label : cipher_input_labels)
  {
    os << "\t{\n";
    os << "\t\tpt := hefloat.NewPlaintext(params, params.MaxLevel())\n";
    os << "\t\tif err := encoder.Encode(values, pt); err != nil { panic(err) }\n";
    os << "\t\tct, err := enc.EncryptNew(pt)\n";
    os << "\t\tif err != nil { panic(err) }\n";
    os << "\t\tencryptedInputs[\"" << label << "\"] = ct\n";
    os << "\t}\n";
  }

  os << R"(
	encryptMillis := float64(time.Since(encryptStart).Microseconds()) / 1000.0
	precisionChecked := false
)";
  if (can_generate_primitive_reference(func))
  {
    gen_primitive_reference_go(func, os);
    os << "\tprecisionChecked = true\n";
  }
  else
  {
    os << "\texpectedOutputs := make(map[string][]float64)\n";
  }
  os << R"(
	computeStart := time.Now()
	for i := 0; i < *repetitions; i++ {
	)";
  os << "\t\t" << func_name;
  os << R"((encryptedInputs, encodedInputs, encryptedOutputs, encodedOutputs, encoder, enc, eval, params)
	}
	computeMillis := float64(time.Since(computeStart).Microseconds()) / 1000.0 / float64(*repetitions)
	decryptStart := time.Now()
	maxAbsError := 0.0
	for name, ct := range encryptedOutputs {
		pt := dec.DecryptNew(ct)
		values := make([]float64, params.MaxSlots())
		if err := encoder.Decode(pt, values); err != nil { panic(err) }
		if precisionChecked {
			for i, expected := range expectedOutputs[name] {
				maxAbsError = math.Max(maxAbsError, math.Abs(values[i]-expected))
			}
		}
		fmt.Printf("%s: [%.4f, %.4f, %.4f, ...]\n", name, values[0], values[1], values[2])
	}
	decryptMillis := float64(time.Since(decryptStart).Microseconds()) / 1000.0
	levelsEnd := -1
	for _, ct := range encryptedOutputs { levelsEnd = ct.Level(); break }
	maxAbsErrorJSON := "null"
	if precisionChecked { maxAbsErrorJSON = fmt.Sprintf("%.12g", maxAbsError) }
	fmt.Printf("BENCHMARK_METRICS {\"repetitions\":%d,\"encrypt_ms\":%.3f,\"compute_ms\":%.3f,\"decrypt_ms\":%.3f,\"galois_keys\":%d,\"bootstrap_count\":)"
     << bootstrap_count
     << R"(,\"level_start\":%d,\"level_end\":%d,\"max_abs_error\":%s,\"precision_checked\":%t}\n",
		*repetitions, encryptMillis, computeMillis, decryptMillis, len(galoisElements), params.MaxLevel(), levelsEnd, maxAbsErrorJSON, precisionChecked)
	_ = encodedOutputs
}
)";
}

} // namespace fheco::code_gen::lattigo

