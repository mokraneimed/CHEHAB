#include "fheco/passes/cse_commut.hpp"
#include "fheco/ir/func.hpp"
#include "fheco/ir/term.hpp"
#include <algorithm>
#include <unordered_set>

using namespace std;

namespace fheco::passes
{
void cse_commut(const shared_ptr<ir::Func> &func)
{
  for (auto id : func->get_top_sorted_terms_ids())
  {
    auto term = func->data_flow().get_term(id);
    if (!term)
      continue;

    if (term->is_leaf())
      continue;

    auto sorted_operands = term->operands();
    sort(sorted_operands.begin(), sorted_operands.end(), ir::Term::EqualPtr{});
    auto new_term = func->insert_op_term(term->op_code(), move(sorted_operands));
    func->replace_term_with(term, new_term);
  }
}
} // namespace fheco::passes