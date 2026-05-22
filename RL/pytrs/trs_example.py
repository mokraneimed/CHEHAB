# main.py

from serializer import expr_to_str
from parser import parse_sexpr
from rules import create_rules
from util import evaluate_expr, generate_random_assignments
from cost import calculate_cost



    

def example_usage():
    
    rules = create_rules(rotations_rules_path='../rotations_rules.txt') # returns a dict of rules {rulename:rule}
    print(rules)
    rule = rules["rotate_16"]

    # Example expression: ( << (Vec a b c d e f g h i ) 1 )
    expr_str = "(VecMinus (VecAdd (VecMinus (VecMul (Vec x_0 1 1 x_3 1 x_5 (- 1 0) 1 1 1 1 1 1 1 x_6 1) (Vec y_0 0 z_2 y_3 0 y_5 0 z_7 0 0 0 0 0 0 y_6 0 0 0 w_2 0 0 0 0 w_7 0 0 0 0 0 0 0 0)) (VecMul (Vec x_0 1 1 x_3 1 x_5 (- 1 0) 1 1 1 1 1 1 1 x_6 1) (<< (Vec y_0 0 z_2 y_3 0 y_5 0 z_7 0 0 0 0 0 0 y_6 0 0 0 w_2 0 0 0 0 w_7 0 0 0 0 0 0 0 0) 16))) (VecMul (Vec 1 1 x_2 1 1 1 1 x_7 1 1 1 1 1 1 1 1) (VecMinus (Vec z_0 0 y_2 z_3 0 z_5 0 y_7 0 0 0 0 (+ (- z_4 w_4) (- 0 (+ 0 (* x_4 y_4)))) 0 z_6 0) (Vec w_0 (+ (* x_1 y_1) (* 1 (- z_1 w_1))) 0 w_3 (- 0 (- 0 (- (- 0 (+ (* x_4 y_4) (- z_4 w_4)))))) w_5 0 0 0 0 0 0 0 0 w_6 0)))) (<< (VecAdd (VecMinus (VecMul (Vec x_0 1 1 x_3 1 x_5 (- 1 0) 1 1 1 1 1 1 1 x_6 1) (Vec y_0 0 z_2 y_3 0 y_5 0 z_7 0 0 0 0 0 0 y_6 0 0 0 w_2 0 0 0 0 w_7 0 0 0 0 0 0 0 0)) (VecMul (Vec x_0 1 1 x_3 1 x_5 (- 1 0) 1 1 1 1 1 1 1 x_6 1) (<< (Vec y_0 0 z_2 y_3 0 y_5 0 z_7 0 0 0 0 0 0 y_6 0 0 0 w_2 0 0 0 0 w_7 0 0 0 0 0 0 0 0) 16))) (VecMul (Vec 1 1 x_2 1 1 1 1 x_7 1 1 1 1 1 1 1 1) (VecMinus (Vec z_0 0 y_2 z_3 0 z_5 0 y_7 0 0 0 0 (+ (- z_4 w_4) (- 0 (+ 0 (* x_4 y_4)))) 0 z_6 0) (Vec w_0 (+ (* x_1 y_1) (* 1 (- z_1 w_1))) 0 w_3 (- 0 (- 0 (- (- 0 (+ (* x_4 y_4) (- z_4 w_4)))))) w_5 0 0 0 0 0 0 0 0 w_6 0)))) 8))"
    
    
    
    a = rule.find_matching_subexpressions(parse_sexpr(expr_str))
    
    
    expr = parse_sexpr(expr_str) # parse from string 
    print("Parsed expression:", expr)  # convert to string
    print("Original expression:", expr_to_str(expr))  # convert to string 
    print("Orignal cost:", calculate_cost(expr))
    """Hadou bech tevalui, rah ygeneri random values for variables li dakhel expression w yhothom fe dict mb3d yb3thom l evaluate_expr"""
    # Create variable assignments for evaluation
    assignments = generate_random_assignments(expr)
    print("Random assignments:", assignments)
    
    # Evaluate original expression
    # original_result = evaluate_expr(expr, assignments)
    # print("Original evaluation:", original_result)

  
    # Find all possible matches where rule can be applied
    matches = rule.find_matching_subexpressions(expr) # returns a list of matches 
    print("matches ",len(matches))
    for match in matches:
        # Apply the rule at each match position
        new_expr = rule.apply_rule(expr, path=match[0])
        
        print("match" ,match)
        print()
        print("lhs" , rule.lhs.expr)
        print()
        
        print("rhs" , rule.rhs)
        print()
        
        print("New expression:", expr_to_str(new_expr))
        
        break
        if new_expr:
            print(f"\nApplied rule: {rule.name}")
            print("New expression:", expr_to_str(new_expr))
            print("New cost:", calculate_cost(new_expr))
            # Verify the transformation preserves semantics
            # new_result = evaluate_expr(new_expr, assignments) 
            # print("New evaluation:", new_result)
            # print("Preserves meaning:", original_result == new_result)

example_usage()