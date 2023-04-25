#include "trs_core.hpp"
#include "ir_utils.hpp"
#include "trs_const.hpp"
#include <iostream>
#include <stdexcept>

using namespace std;

namespace fheco_trs
{
namespace core
{
  std::optional<MatchingMap> match_ir_node(
    std::shared_ptr<ir::Term> ir_node, const MatchingTerm &matching_term, ir::Program *program)
  {
    MatchingMap matching_map;

    if (match_term(ir_node, matching_term, matching_map, program))
      return matching_map;

    return std::nullopt;
  }

  bool match_term(
    std::shared_ptr<ir::Term> ir_node, const MatchingTerm &matching_term, MatchingMap &matching_map,
    ir::Program *program)
  {

    // Order of if statements is important !!!

    // if (ir_node->get_term_type() != fheco_trs::term_type_map[matching_term.get_term_type()])
    // return false;

    // {
    //   auto matching_term_const_value_opt = matching_term.get_value();
    //   if (matching_term_const_value_opt != std::nullopt)
    //   {
    //     auto ir_node_const_value_opt = program->get_entry_value_value(ir_node->get_label());
    //     if (ir_node_const_value_opt == std::nullopt)
    //       return false;

    //     return ir::check_constants_value_equality(
    //       *ir_node_const_value_opt, *matching_term_const_value_opt,
    //       ir_node
    //         ->get_term_type()); // for this to work as expected, TermType passed need to be plaintextType or
    //         scalarType
    //   }
    // }

    auto valid_ir_types = term_type_map[matching_term.get_term_type()];

    if (valid_ir_types.find(ir_node->get_term_type()) == valid_ir_types.end())
      return false;

    auto it = matching_map.find(matching_term.get_term_id());

    if (it != matching_map.end() && it->second != ir_node)
      return false;

    if (matching_term.get_term_type() == TermType::constant)
    {
      auto const_value_opt = program->get_entry_value_value(ir_node->get_label());
      if (const_value_opt == std::nullopt)
        return false;
    }

    if (matching_term.get_opcode() == fheco_trs::OpCode::undefined)
    {
      matching_map[matching_term.get_term_id()] = ir_node;
      return true;
    }

    if (ir_node->get_opcode() != fheco_trs::opcode_mapping[matching_term.get_opcode()])
      return false;

    bool matching_result = true;

    auto matching_term_operands = matching_term.get_operands();
    auto ir_node_operands = ir_node->get_operands();

    if (matching_term_operands.size() != ir_node_operands.size())
      return false;

    for (size_t i = 0; i < matching_term_operands.size(); i++)
    {
      matching_result =
        matching_result && match_term(ir_node_operands[i], matching_term_operands[i], matching_map, program);
      if (matching_result == false)
        break;
    }

    if (matching_result)
      matching_map[matching_term.get_term_id()] = ir_node;

    return matching_result;
  }

  // ir::Number arithmetic_eval(
  //   const MatchingTerm &term, MatchingMap &matching_map, ir::Program *program, FunctionTable &functions_table)
  double arithmetic_eval(
    const MatchingTerm &term, MatchingMap &matching_map, ir::Program *program, FunctionTable &functions_table)
  {
    /*
      This method evaluates arithmetic expressions that involve scalarType and rawDataType
    */

    auto it = functions_table.find(term.get_function_id());

    if (it != functions_table.end())
    {
      auto next_term = it->second(term.get_operands()[0], matching_map, program);
      return arithmetic_eval(next_term, matching_map, program, functions_table);
    }

    if (term.get_opcode() == OpCode::undefined)
    {
      // a constant term, it must be a MatchingTerm with a value or it matches with a scalarType or rawdataType ir node
      if (term.get_value() != std::nullopt)
      {
        return visit(
          ir::overloaded{
            [](auto value_var) -> double { throw std::logic_error("unhandled vector arith eval"); },
            [](const ir::ScalarValue &value_var) -> double {
              return visit(
                ir::overloaded{[](auto value) -> double {
                  return value;
                }},
                value_var);
            }},
          *term.get_value());
        // return ir::get_constant_value_as_number(*term.get_value());
      }
      else
      {
        auto ir_term_itr = matching_map.find(term.get_term_id());
        if (ir_term_itr == matching_map.end())
        {
          throw("arithmetic evaluation impossible");
        }

        if (ir_term_itr->second->get_term_type() == ir::TermType::rawData)
        {
          return std::stod(ir_term_itr->second->get_label());
        }
        else if (ir_term_itr->second->get_term_type() == ir::TermType::scalar)
        {
          ir::ConstantValue constant_value =
            *(*program->get_entry_form_constants_table(ir_term_itr->second->get_label())).get().get_entry_value().value;
          return visit(
            ir::overloaded{
              [](auto value_var) -> double { throw std::logic_error("unhandled vector arith eval"); },
              [](const ir::ScalarValue &value_var) -> double {
                return visit(
                  ir::overloaded{[](auto value) -> double {
                    return value;
                  }},
                  value_var);
              }},
            constant_value);
          // return ir::get_constant_value_as_number(constant_value);
        }
        else
          throw("arithmetic evaluation impossible");
      }
    }
    else
    {
      // for now we consider only binary operations

      if (term.get_operands().size() != 2)
        throw("only binary operations are supported at the moment in arithmetic_eval");

      // ir::Number lhs_value = arithmetic_eval(term.get_operands()[0], matching_map, program, functions_table);
      // ir::Number rhs_value = arithmetic_eval(term.get_operands()[1], matching_map, program, functions_table);
      double lhs_value = arithmetic_eval(term.get_operands()[0], matching_map, program, functions_table);
      double rhs_value = arithmetic_eval(term.get_operands()[1], matching_map, program, functions_table);

      switch (term.get_opcode())
      {
      case OpCode::add:
        return lhs_value + rhs_value;
        break;

      case OpCode::mul:
        return lhs_value * rhs_value;
        break;

      case OpCode::sub:
        return lhs_value - rhs_value;
        break;
      default:
        throw("undefined opcode in arithmetic_eval");
        break;
      }
    }
  }

  bool evaluate_boolean_matching_term(
    const MatchingTerm &matching_term, MatchingMap &matching_map, ir::Program *program, FunctionTable &functions_table)
  {

    // there could be arithmetic expressions but at this level all needs to be evaluated

    if (matching_term.get_opcode() == OpCode::_not)
    {
      auto next_term_to_evaluate = matching_term.get_operands()[0];

      return !evaluate_boolean_matching_term(next_term_to_evaluate, matching_map, program, functions_table);
    }

    auto it = functions_table.find(matching_term.get_function_id());

    if (it != functions_table.end())
    {
      /*
        Here we don't care about calls with type as non booleanType because these concern arithmetic_eval and they are
        leaves so it will be handled below and in arithmetic_eval
      */
      if (matching_term.get_term_type() == TermType::boolean)
      {
        if (matching_term.get_operands().size() == 0)
          throw("missing argument for function to call in evaluate_boolean_matching_term");

        auto next_term_to_evaluate =
          functions_table[matching_term.get_function_id()](matching_term.get_operands()[0], matching_map, program);

        return evaluate_boolean_matching_term(next_term_to_evaluate, matching_map, program, functions_table);
      }
    }

    // if (matching_term.get_opcode() == OpCode::undefined && matching_term.get_term_type() == TermType::boolean)
    // {
    //   auto const_value_opt = matching_term.get_value();
    //   if (const_value_opt == std::nullopt)
    //     throw("boolean leaf term with nullopt value in evaluate_boolean_matching_term");
    //   ir::Number const_value = ir::get_constant_value_as_number(*const_value_opt);
    //   return const_value == 1;
    // }

    if (matching_term.get_opcode() == fheco_trs::OpCode::undefined)
      return true;

    if (
      matching_term.get_opcode() != fheco_trs::OpCode::undefined &&
      matching_term.get_term_type() != fheco_trs::TermType::boolean)
      return false;

    MatchingTerm lhs = matching_term.get_operands()[0];
    MatchingTerm rhs = matching_term.get_operands()[1];

    switch (matching_term.get_opcode())
    {
    case OpCode::_and:
      return evaluate_boolean_matching_term(lhs, matching_map, program, functions_table) &&
             evaluate_boolean_matching_term(rhs, matching_map, program, functions_table);
      break;

    case OpCode::_or:
      return evaluate_boolean_matching_term(lhs, matching_map, program, functions_table) ||
             evaluate_boolean_matching_term(rhs, matching_map, program, functions_table);
      break;
    default:
      break;
    }

    if (rewrite_condition_types.find(lhs.get_term_type()) == rewrite_condition_types.end())
      return false;
    if (rewrite_condition_types.find(rhs.get_term_type()) == rewrite_condition_types.end())
      return false;

    // nodes which arithmetic operands are leaves
    // ir::Number lhs_value, rhs_value;
    double lhs_value, rhs_value;

    lhs_value = arithmetic_eval(lhs, matching_map, program, functions_table);
    rhs_value = arithmetic_eval(rhs, matching_map, program, functions_table);

    switch (matching_term.get_opcode())
    {

    case OpCode::equal:
      return lhs_value == rhs_value;
      break;

    case OpCode::not_equal:
      return lhs_value != rhs_value;
      break;

    case OpCode::less_than:
      return lhs_value < rhs_value;
      break;

    case OpCode::less_than_or_equal:
      return lhs_value <= rhs_value;
      break;

    case OpCode::greater_than:
      return lhs_value > rhs_value;
      break;

    case OpCode::greater_than_or_equal:
      return lhs_value >= rhs_value;
      break;

    default:
      throw("unsuported operand during boolean evaluation");
      break;
    }
  }

  void substitute(
    std::shared_ptr<ir::Term> ir_node, const MatchingTerm &rewrite_rule_rhs, MatchingMap &matching_map,
    ir::Program *program, FunctionTable &functions_table)
  {
    /*
      We call this function after ir_node is matched with lhs of rewrite rule
    */
    std::shared_ptr<ir::Term> new_ir_node =
      make_ir_node_from_matching_term(rewrite_rule_rhs, matching_map, program, functions_table);
    if (new_ir_node->is_operation_node())
    {
      ir_node->rewrite_with_operation(new_ir_node);
    }
    else
    {
      ir_node->delete_operand_term(new_ir_node->get_label());
      const auto node_parents = ir_node->get_parents_labels();
      for (const auto &parent_label : node_parents)
      {
        ir::Term::Ptr parent = program->find_node_in_dataflow(parent_label);
        if (!parent)
          throw logic_error("parent node not found");

        size_t operand_index = parent->delete_operand_term(ir_node->get_label());
        if (operand_index < 0)
          throw logic_error("verified parent however could not delete child operand from parent");

        parent->add_operand(new_ir_node, operand_index);
      }
    }
  }

  std::shared_ptr<ir::Term> make_ir_node_from_matching_term(
    const MatchingTerm &matching_term, MatchingMap &matching_map, ir::Program *program, FunctionTable &functions_table)
  {
    auto it = matching_map.find(matching_term.get_term_id());
    if (it != matching_map.end())
    {
      return it->second;
    }
    // create a new node
    if (matching_term.get_opcode() != fheco_trs::OpCode::undefined)
    {
      if (opcode_mapping.find(matching_term.get_opcode()) == opcode_mapping.end())
        throw("unsuported opcode in ir");

      std::vector<std::shared_ptr<ir::Term>> operands;
      for (auto &m_term_operand : matching_term.get_operands())
      {
        operands.push_back(make_ir_node_from_matching_term(m_term_operand, matching_map, program, functions_table));
      }

      ir::TermType term_type = ir::deduce_ir_term_type(operands);

      std::shared_ptr<ir::Term> ir_node =
        program->insert_operation_node_in_dataflow(opcode_mapping[matching_term.get_opcode()], operands, "", term_type);

      matching_map[matching_term.get_term_id()] = ir_node;

      /*
        This needs to be called here, which means once the ir_node is created, cause basically the function that will
        called it will be applied on an ir node
      */
      if (matching_term.get_function_id() != FunctionId::undefined)
      {
        if (functions_table.find(matching_term.get_function_id()) == functions_table.end())
        {
          throw("undefined function id in make_ir_node_from_matching_term");
        }

        auto new_matching_term = functions_table[matching_term.get_function_id()](matching_term, matching_map, program);

        return make_ir_node_from_matching_term(new_matching_term, matching_map, program, functions_table);
      }

      return ir_node;
    }
    else
    {

      /*
        at this stage it means that wa have a MatchingTerm with a value (a constant), and we will consider it a
        temporary variable in IR
      */
      if (matching_term.get_value() == std::nullopt)
        throw("only constant matching terms are expected at this stage");

      // this will work because constants generated by trs rules are either of type scalar or plaintext
      std::shared_ptr<ir::Term> ir_node =
        std::make_shared<ir::Term>(*fheco_trs::term_type_map[matching_term.get_term_type()].begin());
      /*
        Insert node in dataflow graph
      */
      program->insert_created_node_in_dataflow(ir_node);
      /*
        Insert in constants table
      */
      ir::ConstantTableEntry c_table_entry(
        ir::ConstantTableEntryType::constant, {ir_node->get_label(), *(matching_term.get_value())});
      program->insert_entry_in_constants_table({ir_node->get_label(), c_table_entry});

      return ir_node;
    }
  }

  bool circuit_saving_condition(const ir::Term::Ptr &ir_node)
  {
    return (ir_node->is_operation_node() && ir_node->get_parents_labels().size() <= 1) ||
           (ir_node->is_operation_node() == false);
  }

  bool circuit_saving_condition_rewrite_rule_checker(const MatchingTerm &term, MatchingMap &matching_map)
  {

    {
      auto it = matching_map.find(term.get_term_id());
      if (it == matching_map.end())
        throw("missing matched node in IR in circuit_saving_condition_rewrite_rule_checker");
    }

    for (auto &operand_term : term.get_operands())
    {
      if (operand_term.get_opcode() == fheco_trs::OpCode::undefined)
        continue;
      auto it = matching_map.find(operand_term.get_term_id());
      if (it == matching_map.end())
        throw("missing matched node in IR in circuit_saving_condition_rewrite_rule_checker");

      if (circuit_saving_condition(it->second) == false)
        return false;

      if (circuit_saving_condition_rewrite_rule_checker(operand_term, matching_map) == false)
        return false;
    }

    return true;
  }
} // namespace core
} // namespace fheco_trs