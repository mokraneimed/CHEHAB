#include "fheco/fheco.hpp"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;
using namespace fheco;

int main(int argc, char **argv)
{
  string nom_app = "exemple_present";
  size_t nb_pos = 4;
  int l_bits = 20;
  bool arith_signe = true;
  bool rotation_cycl = false;

  const auto &fonc = Compiler::create_func(nom_app, nb_pos, l_bits, arith_signe, rotation_cycl);

  Ciphertext a("a", -10, 10);
  Ciphertext b("b", -10, 10);
  Ciphertext c("c", -10, 10);
  Ciphertext resultat = a * (b - c) + a * (0 + c);
  resultat.set_output("résultat");

  trs::TRS trs_exemple{trs::Ruleset::toy_ruleset(fonc)};
  trs::print_ruleset(trs_exemple.ruleset(), clog);
  trs_exemple.run(trs::RewriteHeuristic::bottom_up);

  string nom_gen = "gen_he_" + nom_app;
  string chemin_gen = "he/" + nom_gen;
  ofstream flux_entete(chemin_gen + ".hpp");
  ofstream flux_source(chemin_gen + ".cpp");
  Compiler::gen_he_code(fonc, flux_entete, nom_gen + ".hpp", flux_source);

  ofstream flux_exemple_es(nom_app + "_io_example.txt");
  util::print_io_terms_values(fonc, flux_exemple_es);

  ofstream flux_ri(nom_app + "_opt_ir.dot");
  util::draw_ir(fonc, flux_ri, true);

  util::Quantifier quantif(fonc);
  cout << "\ncaractéristiques du circuit optimisé\n";
  quantif.run_all_analysis();
  quantif.print_info(cout);

  return 0;
}