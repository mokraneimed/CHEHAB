from pytrs import parse_sexpr, calculate_cost, Expr, Const, Var, Op,expr_to_str
from param_select import simulate_noise_bfv, NoiseEstimation


expr = parse_sexpr('''(VecAdd (VecMinus (VecAdd (VecMul (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 
2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) 8)) (<< (VecMul (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) 8)) 4)) (<< (VecAdd (VecMul (VecMul (Vec 1 1 v1_0 1 1 1 1 
1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) 8)) (<< (VecMul (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 
1)) 8)) 4)) 2)) (<< (VecMinus (VecAdd (VecMul (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) 8)) (<< (VecMul 
(VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) 8)) 4)) (<< (VecAdd (VecMul (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 
1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) 8)) (<< (VecMul (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) (<< (VecMul (Vec 1 1 v1_0 1 1 1 1 1 1 1 1 v1_1 1 1 1 1) (Vec v1_0 v1_1 v2_0 2 v2_0 v2_1 0 0 1 1 2 v2_1 1 1 1 1)) 
8)) 4)) 2)) 1))
''')
expr2 = parse_sexpr('(Vec ( + ( - ( + v1_0 v2_0 ) ( * ( * v1_0 v2_0 ) 2 ) ) ( - ( + v1_1 v2_1 ) ( * 2 ( * v1_1 v2_1 ) ) ) ) )')
noise_estimates = NoiseEstimation(fresh_noise=68, mul_noise_growth=15, mul_plain_noise_growth=25)
print(simulate_noise_bfv(noise_estimates, expr))