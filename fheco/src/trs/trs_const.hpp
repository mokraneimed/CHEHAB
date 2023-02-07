#pragma once
#include "ir_const.hpp"
#include <unordered_map>
#include <unordered_set>

namespace fheco_trs
{

enum class TermType
{
  rawDataType,
  scalarType,
  ciphertextType,
  plaintextType,
  booleanType,
  opcodeAttribute
};

enum class OpCode
{
  undefined,
  assign,
  encrypt,
  add,
  add_plain,
  mul,
  mul_plain,
  sub,
  sub_plain,
  rotate,
  rotate_rows,
  rotate_columns,
  square,
  negate,
  exponentiate,
  modswitch,
  relinearize,
  rescale,
  less_than_or_equal,
  greater_than_or_equal,
  less_than,
  greater_than,
  not_equal,
  equal,
  _and,
  _or,
  _not
};

inline std::unordered_set<fheco_trs::TermType> rewrite_condition_types = {
  fheco_trs::TermType::rawDataType, fheco_trs::TermType::scalarType};

inline std::unordered_set<fheco_trs::TermType> term_types_attributes = {fheco_trs::TermType::opcodeAttribute};

inline std::unordered_map<fheco_trs::OpCode, ir::OpCode> opcode_mapping = {
  {fheco_trs::OpCode::undefined, ir::OpCode::undefined},
  {fheco_trs::OpCode::add, ir::OpCode::add},
  {fheco_trs::OpCode::mul, ir::OpCode::mul},
  {fheco_trs::OpCode::sub, ir::OpCode::sub},
  {fheco_trs::OpCode::rotate, ir::OpCode::rotate},
  {fheco_trs::OpCode::rotate_rows, ir::OpCode::rotate_rows},
  {fheco_trs::OpCode::add_plain, ir::OpCode::add_plain},
  {fheco_trs::OpCode::sub_plain, ir::OpCode::sub_plain},
  {fheco_trs::OpCode::mul_plain, ir::OpCode::mul_plain},
  {fheco_trs::OpCode::relinearize, ir::OpCode::relinearize},
  {fheco_trs::OpCode::modswitch, ir::OpCode::modswitch},
  {fheco_trs::OpCode::negate, ir::OpCode::negate},
  {fheco_trs::OpCode::square, ir::OpCode::square},
  {fheco_trs::OpCode::rescale, ir::OpCode::rescale},
  {fheco_trs::OpCode::exponentiate, ir::OpCode::exponentiate}};

inline std::unordered_map<fheco_trs::TermType, ir::TermType> term_type_map = {
  {fheco_trs::TermType::ciphertextType, ir::TermType::ciphertextType},
  {fheco_trs::TermType::plaintextType, ir::TermType::plaintextType},
  {fheco_trs::TermType::scalarType, ir::TermType::scalarType},
  {fheco_trs::TermType::rawDataType, ir::TermType::rawDataType}};

} // namespace fheco_trs