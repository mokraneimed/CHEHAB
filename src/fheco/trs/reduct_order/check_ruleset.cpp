#include "fheco/trs/reduct_order/check_ruleset.hpp"
#include "fheco/trs/reduct_order/compon_orders.hpp"
#include "fheco/util/expr_printer.hpp"
#include <iostream>
#include <stdexcept>

using namespace std;

namespace fheco::trs
{

const LexicoProductOrder depth_order{
  {"sum_xdepth"sv, &sum_xdepth_order},
  {"sum_depth"sv, &sum_depth_order},
  {"sum_total_depth"sv, &sum_total_depth_order}};

const LexicoProductOrder ops_cost_order{
  {"mul_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_mul, &is_cipher);
   }},
  {"square_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_square, &is_cipher);
   }},
  {"rotate_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_rotate, &is_cipher);
   }},
  {"mul_plain_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_mul_plain, &is_cipher);
   }},
  {"he_add_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_he_add, &is_cipher);
   }},
  {"sum_rotation_steps"sv, &sum_rotation_steps_order},
  {"cipher_cipher_op_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_cipher_cipher_op, &is_cipher);
   }},
  {"plain_plain_op_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_plain_plain_op, &is_plain);
   }},
  {"const_op_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(rhs, lhs, &is_const_op, &is_plain);
   }},
  {"leaves_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_leaf, &is_var);
   }},
  {"rotate_phi_str"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return phi_str_order(
       lhs, rhs, [](const TermMatcher &term) { return !is_rotate(term); }, &is_rotate);
   }},
  {"const_phi_str"sv, [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return phi_str_order(lhs, rhs, &is_var, &is_const);
   }}};

const LexicoProductOrder joined_order{
  {"sum_xdepth"sv, &sum_xdepth_order},
  {"sum_depth"sv, &sum_depth_order},
  {"sum_total_depth"sv, &sum_total_depth_order},
  {"mul_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_mul, &is_cipher);
   }},
  {"square_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_square, &is_cipher);
   }},
  {"rotate_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_rotate, &is_cipher);
   }},
  {"mul_plain_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_mul_plain, &is_cipher);
   }},
  {"he_add_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_he_add, &is_cipher);
   }},
  {"sum_rotation_steps"sv, &sum_rotation_steps_order},
  {"cipher_cipher_op_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_cipher_cipher_op, &is_cipher);
   }},
  {"plain_plain_op_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_plain_plain_op, &is_plain);
   }},
  {"const_op_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(rhs, lhs, &is_const_op, &is_plain);
   }},
  {"leaves_count"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return class_subterms_count_order(lhs, rhs, &is_leaf, &is_var);
   }},
  {"rotate_phi_str"sv,
   [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return phi_str_order(
       lhs, rhs, [](const TermMatcher &term) { return !is_rotate(term); }, &is_rotate);
   }},
  {"const_phi_str"sv, [](const TermMatcher &lhs, const TermMatcher &rhs) {
     return phi_str_order(lhs, rhs, &is_var, &is_const);
   }}};

void print_line_sep();

void check_ruleset(const Ruleset &ruleset, const LexicoProductOrder &reduct_order, bool throw_on_failure)
{
  clog << "checking ruleset " << ruleset.name() << '\n';
  for (const auto &[root_op_type, rules] : ruleset.rules_by_root_op())
  {
    clog << root_op_type << "_rules:\n";
    for (const auto &rule : rules)
      check_rule(rule, reduct_order, throw_on_failure);
  }
}

void check_rule(const Rule &rule, const LexicoProductOrder &reduct_order, bool throw_on_failure)
{
  clog << util::ExprPrinter::make_rule_str_repr(rule, true, util::ExprPrinter::Mode::infix_expl_paren) << '\n';
  if (rule.has_dynamic_rhs())
  {
    clog << "dynamic rhs, invariants needed\n";
    return;
  }

  const auto &lhs = rule.lhs();
  auto rhs = rule.get_rhs();
  for (const auto &[name, order] : reduct_order)
  {
    clog << "trying order " << name << ": ";
    auto comp_result = order(lhs, rhs);
    if (comp_result == CompResult::less)
    {
      clog << "rejected\n";
      if (throw_on_failure)
        throw logic_error("rejected rule");
    }
    else if (comp_result == CompResult::not_generalizable)
    {
      clog << "order cannot be generalized, needs condition of application\n";
      if (rule.has_cond())
        clog << "checking the correctness of the existing condition is not supported\n";
      else
        clog << "condition is not provided\n";
    }
    else if (comp_result == CompResult::equal)
    {
      clog << "undefined order\n";
      continue;
    }
    else
      clog << "accepted\n";

    print_line_sep();
    return;
  }
  clog << "unordered\n";
  if (throw_on_failure)
    throw logic_error("unordered rule");
}

void print_line_sep()
{
  clog << std::string(100, '-') << '\n';
}
} // namespace fheco::trs