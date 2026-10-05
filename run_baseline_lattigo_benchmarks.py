import os
import subprocess 
import csv 
import re
import statistics

benchmarks_folder = "benchmarks"  
build_folder = os.path.join("build", "benchmarks")

# Lattigo operation names for counting in generated Go code
lattigo_operations = ["AddNew", "SubNew", "MulNew", "RotateNew", "NegNew", "MulRelinNew"]
# CSV column names matching the SEAL version for comparability
operations = ["add", "sub", "multiply_plain", "rotate_rows", "negate", "multiply"]
# Mapping: Lattigo op name -> CSV column name
lattigo_to_csv_op = {
    "AddNew": "add",
    "SubNew": "sub",
    "MulNew": "multiply_plain",
    "RotateNew": "rotate_rows",
    "NegNew": "negate",
    "MulRelinNew": "multiply",
}

infos = ["benchmark"]
additional_infos = ["Depth", "Multiplicative Depth", "compile_time (s)", "circuit_execution_time (s)",
                    'galois_keys_generation_time (s)', 'total_execution_time (s)',
                    'rotation_keys_size (MB)']
infos.extend(operations)
infos.extend(additional_infos)

try:
    print("run=> cmake', '-S', '.', '-B', 'build' ")
    result = subprocess.run(
        ['cmake', '-S', '.', '-B', 'build'], 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )
    print("run=> 'cmake', '--build', 'build'")
    result = subprocess.run(
        ['cmake', '--build', 'build'], 
        check=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        universal_newlines=True
    )  
except subprocess.CalledProcessError as e:
    print(f"Command failed with error:\n{e.stderr}")   

benchmark_folders = ["dot_product"]
exceptions = ["max", "sort", "discrete_cosin_transform", "poly_derivative"]
benchmarks_slot_counts = {
    "max": [3, 4, 5],
    "sort": [3],
    "discrete_cosin_transform": [1],
    "poly_derivative": [1]
}

optimization_method = 1
cse_enabled = 1
vectorize_code = 1
slot_counts = [4, 8]
iterations = 2
window_size = 0
depths = [5]
regimes = ["100-50"]
number_instances_each_polynomial_configuration = 1
compile_time_timeout_seconds = 7200
go_run_timeout_seconds = 600
output_csv = "results_baseline_lattigo.csv"


with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(infos)


# ── helpers ────────────────────────────────────────────────────────────────────

def ensure_go_module(build_path):
    """Initialize Go module and fetch dependencies if not already done."""
    go_mod_path = os.path.join(build_path, "go.mod")
    if not os.path.exists(go_mod_path):
        print("  [go] Initializing Go module...")
        subprocess.run(
            ['go', 'mod', 'init', 'benchmark_fhe'],
            cwd=build_path,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True
        )
    print("  [go] Running go mod tidy...")
    subprocess.run(
        ['go', 'mod', 'tidy'],
        cwd=build_path,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True
    )


def run_benchmark(subfolder_name, slot_count, build_path, build_path_he, build_path_he_build):
    """
    Run one (subfolder_name, slot_count) point using Lattigo backend with the baseline agent.
    Returns row or None on timeout.
    """
    benchmark_compilation_timed_out = False
    operation_stats = {
        "add": [], "sub": [], "multiply_plain": [], "rotate_rows": [],
        "negate": [], "multiply": [], "Depth": [], "Multiplicative Depth": [],
        "compile_time (s)": [], "circuit_execution_time (s)": [], "galois_keys_generation_time (s)": [],
        "total_execution_time (s)": [], "rotation_keys_size (MB)": [],
    }

    if not subfolder_name in exceptions:
        pro = subprocess.Popen(['python3', 'generate_{}.py'.format(subfolder_name), '--slot_count', str(slot_count)], cwd=build_path)
        pro.wait()

    for iteration in range(iterations):
        print(f"===> Running iteration : {iteration + 1}")
        # backend=1 triggers Lattigo Go code generation; framework=baseline uses the baseline agent
        benchmark_run_command = f"./{subfolder_name} {vectorize_code} {slot_count} baseline {optimization_method} {window_size} 1 {cse_enabled} 1 1"
        try:
            result = subprocess.run(
                benchmark_run_command, shell=True, check=False,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, cwd=build_path,
                timeout=compile_time_timeout_seconds
            )
            lines = result.stdout.splitlines()
            compile_time_found = False
            poly_mod_found = True
            for line in lines:
                if ' ms' in line:
                    optimization_time = float(line.split()[0])
                    operation_stats["compile_time (s)"].append(optimization_time)
                    compile_time_found = True
                if 'poly_mod:' in line:
                    print(f"======> poly_mod : {line}")
                    poly_mod = float(line.split()[1])
                    poly_mod_found = True
                if compile_time_found and poly_mod_found:
                    break
            depth_match = re.search(r'max:\s*\((\d+),\s*(\d+)\)', result.stdout)
            depth = int(depth_match.group(1)) if depth_match else None
            multiplicative_depth = int(depth_match.group(2)) if depth_match else None
            print(f"Depth=>{depth}, multiplicative_depth=>{multiplicative_depth}")
            operation_stats["Depth"].append(depth)
            operation_stats["Multiplicative Depth"].append(multiplicative_depth)
        except subprocess.TimeoutExpired:
            print(f"Command `{benchmark_run_command}` timed out after {compile_time_timeout_seconds} seconds.")
            benchmark_compilation_timed_out = True
        except subprocess.CalledProcessError as e:
            error_message = e.stderr if e.stderr else "No error message available."
            print("Command for {} failed with error:\n{}".format(subfolder_name, error_message))
            continue

        if benchmark_compilation_timed_out:
            break

        he_build_ok = (result.returncode == 0)
        if he_build_ok:
            # ── Lattigo: run Go code instead of building C++ ──────────────────
            go_file = os.path.join(build_path, "generated_fhe.go")
            if os.path.exists(go_file):
                ensure_go_module(build_path)

                if iteration == iterations - 1:
                    try:
                        for counter in range(iterations):
                            go_result = subprocess.run(
                                ['go', 'run', 'generated_fhe.go'],
                                cwd=build_path,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True,
                                timeout=go_run_timeout_seconds
                            )
                            print("**Lattigo fhe run done**")
                            if counter > 0 or iterations == 1:
                                go_lines = go_result.stdout.splitlines()
                                print(f"returned lines : \n {go_lines} \n\n")
                                for line in go_lines:
                                    if 'circuit_execution_time_(ms):' in line:
                                        operation_stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                                    if 'galois_keys_generation_time_(ms):' in line:
                                        operation_stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                                    if 'total_execution_time_(ms):' in line:
                                        operation_stats["total_execution_time (s)"].append(float(line.split()[1]))
                                    if 'rotation_keys_size_(MB):' in line:
                                        operation_stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                    except subprocess.TimeoutExpired:
                        print(f"Go run timed out after {go_run_timeout_seconds} seconds.")
                    except Exception as e:
                        print(f"Failed running Lattigo code for benchmark: {subfolder_name}: {e}")
            else:
                print(f"  [warn] generated_fhe.go not found at {go_file}")

        # Count operations from generated Go file
        go_gen_file = os.path.join(build_path, "generated_fhe.go")
        if os.path.exists(go_gen_file):
            with open(go_gen_file, "r") as file:
                file_content = file.read()
                for lattigo_op, csv_op in lattigo_to_csv_op.items():
                    nb_occurrences = len(re.findall(rf'\b{lattigo_op}', file_content))
                    operation_stats[csv_op].append(int(nb_occurrences))

    # ── build row ──────────────────────────────────────────────────────────────
    bench_name = subfolder_name + "_" + str(slot_count)
    row = [bench_name]

    if not benchmark_compilation_timed_out:
        for key, values in operation_stats.items():
            if values == []:
                print(f"Warning: No values found for {key} in {subfolder_name} with slot_count {slot_count}.")
                result = "N/A"
            else:
                result = statistics.median(values)
                if key in ["compile_time (s)", "circuit_execution_time (s)",
                           "galois_keys_generation_time (s)", "total_execution_time (s)"]:
                    result = result / 1000
                    result = format(result, ".3f")
                row.append(result)
            print(f"{key} {values} {result}")

    # ── write immediately ────────────────────────────────────────────────────
    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    if benchmark_compilation_timed_out:
        return None

    return row


# ── main loop ─────────────────────────────────────────────────────────────────
for subfolder_name in benchmark_folders:
    benchmark_path = os.path.join(benchmarks_folder, subfolder_name)
    build_path = os.path.join(build_folder, subfolder_name)
    if os.path.isdir(build_path):
        updated_slot_counts = slot_counts
        if subfolder_name in exceptions:
            updated_slot_counts = benchmarks_slot_counts[subfolder_name]

        for slot_count in updated_slot_counts:
            build_path_he = os.path.join(build_path, "he")
            build_path_he_build = os.path.join(build_path_he, "build")

            print("****************************************************************")
            print(f"*****run {subfolder_name} , for slot_count : {slot_count}******")
            try:
                row = run_benchmark(subfolder_name, slot_count,
                                    build_path, build_path_he, build_path_he_build)
            except Exception as e:
                print(f"Command for {subfolder_name} failed with error:\n{e}")
                continue

#################################################################################################
# ── poly-tree helpers ──────────────────────────────────────────────────────────

def run_poly_benchmark(subfolder_name, build_path, build_path_he, build_path_he_build,
                       benchmark_name, tree_depth, instance, regime):
    """
    Run one (benchmark_name) point for polynomial trees using Lattigo with the baseline agent.
    Returns row or None on timeout.
    """
    benchmark_compilation_timed_out = False
    operation_stats = {
        "add": [], "sub": [], "multiply_plain": [], "rotate_rows": [],
        "negate": [], "multiply": [], "Depth": [], "Multiplicative Depth": [],
        "compile_time (s)": [], "circuit_execution_time (s)": [],
        "galois_keys_generation_time (s)": [], "total_execution_time (s)": [],
        "rotation_keys_size (MB)": [],
    }

    for iteration in range(iterations):
        optimization_time = ""
        execution_time = ""
        depth = ""
        multiplicative_depth = ""

        if not os.path.isdir(build_path):
            continue

        print(f"=========> Iteration : {iteration + 1}")
        # backend=1 triggers Lattigo Go code generation; framework=baseline
        command = (f"./{subfolder_name} {tree_depth} {instance} {regime} "
                   f"{vectorize_code} {optimization_method} {window_size} "
                   f"1 {cse_enabled} 1 1")
        try:
            result = subprocess.run(
                command, shell=True, check=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, cwd=build_path,
                timeout=compile_time_timeout_seconds
            )
            lines = result.stdout.splitlines()
            compile_time_found = False
            poly_mod_found = True
            for line in lines:
                if ' ms' in line:
                    optimization_time = float(line.split()[0])
                    operation_stats["compile_time (s)"].append(optimization_time)
                    compile_time_found = True
                if 'poly_mod:' in line:
                    print(f"======> poly_mod : {line}")
                    poly_mod_found = True
                if compile_time_found and poly_mod_found:
                    break

            depth_match = re.search(r'max:\s*\((\d+),\s*(\d+)\)', result.stdout)
            depth = int(depth_match.group(1)) if depth_match else None
            multiplicative_depth = int(depth_match.group(2)) if depth_match else None
            print(f"Depth: {depth} -- MultiplicativeDepth: {multiplicative_depth}")
            operation_stats["Depth"].append(depth)
            operation_stats["Multiplicative Depth"].append(multiplicative_depth)

        except subprocess.TimeoutExpired:
            print(f"Command `{command}` timed out after {compile_time_timeout_seconds} seconds.")
            benchmark_compilation_timed_out = True
        except subprocess.CalledProcessError as e:
            print(f"Command for {subfolder_name} failed with error:\n{e.stderr}")

        if benchmark_compilation_timed_out:
            break

        # ── Lattigo: run Go code ──────────────────────────────────────────────
        go_file = os.path.join(build_path, "generated_fhe.go")
        if os.path.exists(go_file):
            ensure_go_module(build_path)

            if iteration == iterations - 1:
                try:
                    for counter in range(iterations):
                        go_result = subprocess.run(
                            ['go', 'run', 'generated_fhe.go'],
                            cwd=build_path,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True,
                            timeout=go_run_timeout_seconds
                        )
                        print("**Lattigo fhe run done**")
                        if counter > 0 or iterations == 1:
                            go_lines = go_result.stdout.splitlines()
                            print(f"returned lines : \n {go_lines} \n\n")
                            for line in go_lines:
                                if 'circuit_execution_time_(ms):' in line:
                                    operation_stats["circuit_execution_time (s)"].append(float(line.split()[1]))
                                if 'galois_keys_generation_time_(ms):' in line:
                                    operation_stats["galois_keys_generation_time (s)"].append(float(line.split()[1]))
                                if 'total_execution_time_(ms):' in line:
                                    operation_stats["total_execution_time (s)"].append(float(line.split()[1]))
                                if 'rotation_keys_size_(MB):' in line:
                                    operation_stats["rotation_keys_size (MB)"].append(float(line.split()[1]))
                except subprocess.TimeoutExpired:
                    print(f"Go run timed out after {go_run_timeout_seconds} seconds.")
                except Exception as e:
                    print(f"Failed running Lattigo code for benchmark: {subfolder_name}: {e}")

        # Count operations from generated Go file
        go_gen_file = os.path.join(build_path, "generated_fhe.go")
        if os.path.exists(go_gen_file):
            with open(go_gen_file, "r") as f:
                file_content = f.read()
                for lattigo_op, csv_op in lattigo_to_csv_op.items():
                    nb_occurrences = len(re.findall(rf'\b{lattigo_op}', file_content))
                    operation_stats[csv_op].append(int(nb_occurrences))

    # ── build row ──────────────────────────────────────────────────────────────
    row = [benchmark_name]

    if not benchmark_compilation_timed_out:
        for key, values in operation_stats.items():
            if not values:
                print(f"Warning: No values found for {key} in {benchmark_name}.")
                result_val = "N/A"
            else:
                result_val = statistics.median(values)
                if key in ["compile_time (s)", "circuit_execution_time (s)",
                           "galois_keys_generation_time (s)", "total_execution_time (s)"]:
                    result_val = result_val / 1000
                    result_val = format(result_val, ".3f")
            row.append(result_val)
            print(f"{key} {values} {result_val}")

    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    if benchmark_compilation_timed_out:
        return None

    return row


# ── poly-tree main loop ────────────────────────────────────────────────────────
print("Run polynomial benchmarks (Lattigo baseline) !!!!!!")
polynomial_folders = ["polynomials_coyote"]

for subfolder_name in polynomial_folders:
    build_path = os.path.join(build_folder, subfolder_name)
    build_path_he = os.path.join(build_path, "he")
    build_path_he_build = os.path.join(build_path_he, "build")

    for regime in regimes:
        for tree_depth in depths:
            for instance in range(1, number_instances_each_polynomial_configuration + 1):
                benchmark_name = f'tree_{regime}_{tree_depth}_{instance}'
                print(f"\nBenchmark '{benchmark_name}' will be run...")

                print("*" * 64)
                try:
                    row = run_poly_benchmark(
                        subfolder_name, build_path, build_path_he, build_path_he_build,
                        benchmark_name, tree_depth, instance, regime
                    )
                except Exception as e:
                    print(f"Command for {benchmark_name} failed with error:\n{e}")
                    continue
