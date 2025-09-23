from params import EncParams
from noise_simulation import simulate_noise_bfv
from bfv_values import security_standard, bfv_noise_estimates_seal, SecurityLevel



MOD_BIT_COUNT_MAX = 60
MAX_N = 32768

def select_parameters_bfv(plain_mod_size, slot_count, expr, sec_level=None):
    while (plain_mod_size not in bfv_noise_estimates_seal 
            and plain_mod_size < MOD_BIT_COUNT_MAX):
        plain_mod_size += 1


    #check if key exists
    if plain_mod_size not in bfv_noise_estimates_seal:
        raise ValueError(
            "noise estimates maximum plaintext modulus size smaller than bit_width+1+signedness"
        )

    #check if value (per-n map) is empty
    per_poly_mod_estimates = bfv_noise_estimates_seal[plain_mod_size]
    if not per_poly_mod_estimates:
        raise ValueError(
            "empty per polynomial modulus degree estimates map for the given plaintext modulus size"
        )

    poly_mod_degree = 2
    while (poly_mod_degree < slot_count):
        poly_mod_degree = poly_mod_degree*2

    poly_mod_degree = poly_mod_degree*2

    while (all(poly_mod_degree != n for n, _ in per_poly_mod_estimates)
            and poly_mod_degree < MAX_N):
            poly_mod_degree = poly_mod_degree*2

    if all(poly_mod_degree != n for n, _ in per_poly_mod_estimates):
        raise ValueError(
            "the maximum polynomial modulus degree of the noise estimates for the given plaintext modulus "
                        "size is smaller than vector_size"
        )

    circuit_noise = 0
    params = EncParams()

    while not all(poly_mod_degree != n for n, _ in per_poly_mod_estimates):
        per_poly_noise = next((item for item in per_poly_mod_estimates if item[0] == poly_mod_degree))
        noise_estimation_value = per_poly_noise[1]
        circuit_noise = simulate_noise_bfv(noise_estimation_value, expr)
        coeff_mod_size = plain_mod_size + circuit_noise
        params.set_params(poly_mod_degree, coeff_mod_size, plain_mod_size)

        if sec_level is None:
            break

        if sec_level not in security_standard:
            raise ValueError("unknown security level")

        sec_level_standard_it = security_standard[sec_level]

        nq_value = next((nq for nq in sec_level_standard_it if nq.n == poly_mod_degree), None)

        if nq_value is None:
            raise ValueError(
                "No NQValue object found with the given polynomial modulus degree"
            )

        if params.q <= nq_value.q:
            break

        poly_mod_degree = poly_mod_degree*2

    if all(poly_mod_degree != n for n, _ in per_poly_mod_estimates):
        raise ValueError(
            "No valid parameters found for the given vector size and plaintext modulus bit width"
        )

    if sec_level is not None:
        security_standard_it = security_standard.get(sec_level, None)
        if security_standard_it is None:
            raise ValueError("unknown security level")
        nq_value = next((nq for nq in security_standard_it if nq.n == poly_mod_degree), None)
        if nq_value is None:
            raise ValueError(
                "No NQValue object found with the given polynomial modulus degree"
            )
        max_coeff_mod_size = nq_value.q
        params.increase_q(max_coeff_mod_size - params.q)
    else:
        params.increase_q(MOD_BIT_COUNT_MAX)

    return params                                     