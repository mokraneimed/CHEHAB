#include "fheco/ir/func.hpp"
#include "fheco/ir/term.hpp"
#include "fheco/passes/reduce_rotation_keys.hpp"
#include "fheco/util/common.hpp"
#include <algorithm>
#include <cstdlib>
#include <functional>
#include <stdexcept>

using namespace std;

namespace fheco::passes
{

/* ======================= SINGLE PASS ======================= */

static unordered_set<int> reduce_rotation_keys_single_pass(
    const shared_ptr<ir::Func> &func,
    size_t keys_threshold)
{
  unordered_map<int, int> steps_freq;
  ir::Term::PtrSet decomp_candidate_terms;

  /* 🔑 CRITICAL FIX:
     Re-scan the IR every pass to see newly created rotations */
  for (auto term : func->get_top_sorted_terms())
  {
    if (term->type() == ir::Term::Type::cipher &&
        term->op_code().type() == ir::OpCode::Type::rotate)
    {
      int steps = term->op_code().steps();
      ++steps_freq[steps];
      decomp_candidate_terms.insert(func->data_flow().get_term(term->id()));
    }
  }

  vector<int> ordered_used_steps;
  ordered_used_steps.reserve(steps_freq.size());
  for (const auto &e : steps_freq)
    ordered_used_steps.push_back(e.first);

  int keys_count = ordered_used_steps.size();
  if (keys_count <= keys_threshold)
    return unordered_set<int>(ordered_used_steps.begin(), ordered_used_steps.end());

  unordered_map<int, vector<int>> steps_nafs;
  for (auto steps : ordered_used_steps)
  {
    if (util::is_power_of_two(abs(steps)) && abs(steps) > 1)
    {
      int half = steps / 2;
      steps_nafs.emplace(steps, vector<int>{half, half});
    }
    else
    {
      steps_nafs.emplace(steps, naf(steps));
    }
  }

  unordered_map<int, int> steps_costs;
  for (const auto &e : steps_freq)
  {
    auto steps = e.first;
    auto freq = e.second;
    steps_costs.emplace(steps, freq * (steps_nafs.at(steps).size() - 1));
  }

  sort(ordered_used_steps.begin(), ordered_used_steps.end(),
       [&steps_costs, &steps_nafs](int lhs, int rhs) {
         auto lhs_cost = steps_costs.at(lhs);
         auto rhs_cost = steps_costs.at(rhs);
         if (lhs_cost == rhs_cost)
           return steps_nafs.at(lhs) < steps_nafs.at(rhs);
         return lhs_cost > rhs_cost;
       });

  unordered_set<int> steps_to_decomp;
  unordered_set<int> used_steps;

  int init_steps_count = ordered_used_steps.size();
  for (int i = 0; i < init_steps_count; ++i)
  {
    auto min_cost_steps = ordered_used_steps.back();
    ordered_used_steps.pop_back();

    steps_to_decomp.insert(min_cost_steps);
    keys_count -= 1;

    for (auto comp : steps_nafs.at(min_cost_steps))
    {
      if (used_steps.insert(comp).second)
        keys_count += 1;
    }

    if (keys_count <= keys_threshold)
      break;
  }

  // if (keys_count > keys_threshold)
  //   throw logic_error(
  //       "could not go lower than the threshold, maybe keys_threshold is invalid (too low)");

  for (auto s : ordered_used_steps)
    used_steps.insert(s);

  for (auto s : steps_to_decomp)
    sort(steps_nafs.at(s).begin(), steps_nafs.at(s).end(), greater<int>());

  for (auto term : decomp_candidate_terms)
  {
    auto steps = term->op_code().steps();
    if (steps_to_decomp.count(steps))
      decomp_rotation_term(func, term, steps_nafs.at(steps));
  }

  return used_steps;
}

/* ======================= ITERATIVE DRIVER ======================= */

unordered_set<int> reduce_rotation_keys(
    const shared_ptr<ir::Func> &func,
    size_t keys_threshold)
{
  unordered_set<int> prev_steps;
  unordered_set<int> curr_steps;

  while (true)
  {
    curr_steps = reduce_rotation_keys_single_pass(func, keys_threshold);

    if (curr_steps.size() <= keys_threshold)
      return curr_steps;

    if (curr_steps == prev_steps)
      break; // no progress → stop

    prev_steps = curr_steps;
  }

  throw logic_error(
      "could not go lower than the threshold after iterative reduction");
}

/* ======================= ROTATION DECOMP ======================= */

void decomp_rotation_term(
    const shared_ptr<ir::Func> &func,
    ir::Term *term,
    const vector<int> &steps_seq)
{
  if (term->op_code().type() != ir::OpCode::Type::rotate)
    throw invalid_argument("term must be a rotation term");

  auto arg = term->operands()[0];
  for (auto steps : steps_seq)
  {
    if (steps == 0 || abs(steps) == func->slot_count())
      continue;

    auto rotation_comp =
        func->insert_op_term(ir::OpCode::rotate(steps), {arg});
    arg = rotation_comp;
  }

  if (*arg != *term->operands()[0])
    func->replace_term_with(term, arg);
}

/* ======================= NAF ======================= */

vector<int> naf(int value)
{
  vector<int> res;
  bool sign = value < 0;
  value = abs(value);

  for (int i = 0; value; i++)
  {
    int zi = (value & 1) ? 2 - (value & 3) : 0;
    value = (value - zi) >> 1;
    if (zi)
      res.push_back((sign ? -zi : zi) * (1 << i));
  }

  return res;
}

} // namespace fheco::passes
