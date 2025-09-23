from pytrs import calculate_cost, parse_sexpr, Expr, Var, Const, Op
from utils import load_expressions, topological_sort
from enum import Enum
from bfv_values import NoiseEstimation

TermType = Enum('TermType', 'cipher plain')

def term_type(term:Expr):
    if isinstance(term, Const):
        return TermType.plain
    if isinstance(term, Var):
        return TermType.cipher
    if isinstance(term, Op):
        for arg in term.args:
            if term_type(arg) == TermType.cipher:
                return TermType.cipher
        return TermType.plain
    raise ValueError("unknown term type")

def simulate_noise_bfv(noise_estimates:NoiseEstimation, exp:Expr):
    fresh_noise = noise_estimates.fresh_noise
    mul_noise_growth = noise_estimates.mul_noise_growth
    mul_plain_noise_growth = noise_estimates.mul_plain_noise_growth
    nodes_noise = {}

    operations_noise_growth = {
    ('*', TermType.cipher, TermType.cipher): mul_noise_growth,
    ('*', TermType.cipher, TermType.plain): mul_plain_noise_growth,
    ('*', TermType.plain, TermType.cipher): mul_plain_noise_growth,
    ('VecMul', TermType.cipher, TermType.cipher): mul_noise_growth,
    ('VecMul', TermType.cipher, TermType.plain): mul_plain_noise_growth,
    ('VecMul', TermType.plain, TermType.cipher): mul_plain_noise_growth,
    ('+', TermType.cipher, TermType.cipher): 1,
    ('+', TermType.cipher, TermType.plain): 0,
    ('+', TermType.plain, TermType.cipher): 0,
    ('VecAdd', TermType.cipher, TermType.cipher): 1,
    ('VecAdd', TermType.cipher, TermType.plain): 0,
    ('VecAdd', TermType.plain, TermType.cipher): 0,
    ('-', TermType.cipher, TermType.cipher): 1,
    ('-', TermType.cipher, TermType.plain): 0,
    ('-', TermType.plain, TermType.cipher): 0,
    ('VecMinus', TermType.cipher, TermType.cipher): 1,
    ('VecMinus', TermType.cipher, TermType.plain): 0,
    ('VecMinus', TermType.plain, TermType.cipher): 0,
    ('-', TermType.cipher, None): 1,
    ('VecNeg', TermType.cipher, None): 1,
    ('<<', TermType.cipher, TermType.plain): 1,
    ('encrypt', TermType.plain, None): fresh_noise,
    ('Vec'): 0
    }

    circuit_noise = fresh_noise

    sorted_terms = topological_sort(exp)
    for term in sorted_terms:
        if term_type(term) != TermType.cipher:
            if isinstance(term, Op) and term.op != "Vec":
                continue
        if term in nodes_noise:
            raise ValueError("repeated node in dataflow_sorted_nodes")
        if not isinstance(term, Op):
            nodes_noise[term] = fresh_noise
        else:
            result_noise = 0
            if term.op == "Vec":
                for arg in term.args:
                    arg_noise = 0
                    if term_type(arg) == TermType.cipher or (isinstance(arg, Op) and term.op == "Vec"):   
                        arg_it = nodes_noise.get(arg, None)
                        if arg_it is None:
                            raise ValueError("parent handled before child")
                        arg_noise = arg_it
                    if arg_noise > result_noise:
                        result_noise = arg_noise
                result_noise += operations_noise_growth[('Vec')] 
            elif len(term.args) == 2:
                arg1 = term.args[0]
                arg2 = term.args[1]

                is_cipher = False

                arg1_noise = 0
                arg1_it = nodes_noise.get(arg1, None)       
                if term_type(arg1) == TermType.cipher or (isinstance(arg1, Op) and arg1.op == "Vec"):   
                    if arg1_it is None:
                        raise ValueError("parent handled before child")
                    is_cipher = True
                    arg1_noise = arg1_it
                arg2_noise = 0
                arg2_it = nodes_noise.get(arg2, None)
                if term_type(arg2) == TermType.cipher or (isinstance(arg2, Op) and arg2.op == "Vec"):
                    if arg2_it is None:
                        raise ValueError("parent handled before child")
                    is_cipher = True
                    arg2_noise = arg2_it

                if not is_cipher:
                    raise ValueError("binary operation ciphertext node with two non-ciphertext operands")

                noise_growth_it = 0
                if (term.op, term_type(arg1), term_type(arg2)) in operations_noise_growth:
                    noise_growth_it = operations_noise_growth[(term.op, term_type(arg1), term_type(arg2))]
                result_noise = max(arg1_noise, arg2_noise) + noise_growth_it
            elif len(term.args) == 1:
                arg1 = term.args[0]
                arg1_noise = 0
                if term_type(arg1) != TermType.cipher: 
                    if isinstance(arg1, Op) and arg1.op != "Vec":
                        if term.op != "encrypt":
                            raise ValueError("unary operation ciphertext node with non-ciphertext operand")
                        arg1_noise = 0
                else:
                    arg1_it = nodes_noise.get(arg1, None)
                    if arg1_it is None:
                        raise ValueError("parent handled before child")
                    arg1_noise = arg1_it

                noise_growth_it = 0
                if (term.op, term_type(arg1), None) in operations_noise_growth:
                    noise_growth_it = operations_noise_growth[(term.op, term_type(arg1), None)]
                result_noise = arg1_noise + noise_growth_it

            nodes_noise[term] = result_noise
            if result_noise > circuit_noise:
                circuit_noise = result_noise                           
    return circuit_noise            

