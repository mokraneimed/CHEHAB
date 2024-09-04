import os
import shutil
import subprocess
import csv
import re

# Specify the parent folder containing the benchmarks and build subfolders
benchmarks_folder = "benchmarks"
build_folder = os.path.join("build", "benchmarks")
output_csv = "results.csv"
vectorization_csv = "vectorization.csv"

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Benchmark", "Compilation Time", "Depth",
                      "Multiplicative Depth", "Plain Multiplications",
                        "Multiplications","Additions","Rotations","Squares"])
    
with open(vectorization_csv, mode = 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Benchmark","Vecorization Time"])    
    
try:
    result = subprocess.run(['cmake','-G','MSYS Makefiles','-S','.','-B','build'], shell=True, check=True, capture_output=True, text=True)
    result = subprocess.run(['cmake','--build','build'], shell=True, check=True, capture_output=True, text=True)
    result = subprocess.run(['cmake','--install','build'], shell=True, check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with error:\n{e.stderr}")    

# Iterate through each item in the benchmarks folder
for subfolder_name in os.listdir(benchmarks_folder):
    benchmark_path = os.path.join(benchmarks_folder, subfolder_name)
    build_path = os.path.join(build_folder, subfolder_name)

    if os.path.isdir(build_path):
        # Paths for the .cpp files
        benchmark_cpp_path = os.path.join(benchmark_path, f"{subfolder_name}.cpp")
        backup_cpp_path = os.path.join(benchmark_path, "backup.cpp")
        fhe_generated_cpp_path = os.path.join(build_path, "fhe_vectorized.cpp")

        # Step 1: Construct the command to run `.\\benchmark` in the subfolder
        command = f".\\{subfolder_name} 1 0"
        # print(f"Running command in {build_path}: {command}")

        # # Set the current working directory to the subfolder
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=build_path)
            lines = result.stdout.splitlines()
            for line in lines:
                if 'ms' in line:
                    time_str = line.split()[0]
                    break
            with open(vectorization_csv, mode = 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([subfolder_name, time_str])   
            # print(f"Output for {subfolder_name}:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Command for {subfolder_name} failed with error:\n{e.stderr}")

        # Step 2: Copy the content of $benchmark.cpp to backup.cpp
        try:
            shutil.copyfile(benchmark_cpp_path, backup_cpp_path)
            # print(f"Copied content of {benchmark_cpp_path} to {backup_cpp_path}")
        except FileNotFoundError:
            print(f"File {benchmark_cpp_path} not found, skipping {subfolder_name}")
            continue

        # Step 3: Replace the content of fhe_generated.cpp with $benchmark.cpp
        try:
            with open(fhe_generated_cpp_path, 'r') as src_file:
                fhe_generated_content = src_file.read()

            with open(benchmark_cpp_path, 'w') as dest_file:
                dest_file.write(fhe_generated_content)

            # print(f"Replaced content of {benchmark_cpp_path} with {fhe_generated_cpp_path}")
        except FileNotFoundError:
            print(f"File {fhe_generated_cpp_path} not found in {build_path}, skipping {subfolder_name}")
            continue

# Step 4: compile the vectorized code.
try:
    result = subprocess.run(['cmake','-G','MSYS Makefiles','-S','.','-B','build'], shell=True, check=True, capture_output=True, text=True)
    result = subprocess.run(['cmake','--build','build'], shell=True, check=True, capture_output=True, text=True)
    result = subprocess.run(['cmake','--install','build'], shell=True, check=True, capture_output=True, text=True)
except subprocess.CalledProcessError as e:
    print(f"Command failed with error:\n{e.stderr}")

for subfolder_name in os.listdir(benchmarks_folder):
    benchmark_path = os.path.join(benchmarks_folder, subfolder_name)
    build_path = os.path.join(build_folder, subfolder_name)

    if os.path.isdir(build_path):
        benchmark_cpp_path = os.path.join(benchmark_path, f"{subfolder_name}.cpp")
        backup_cpp_path = os.path.join(benchmark_path, "backup.cpp")

        try:
            shutil.copyfile(backup_cpp_path, benchmark_cpp_path)
            os.remove(backup_cpp_path)
            # print(f"Copied content of {benchmark_cpp_path} to {backup_cpp_path}")
        except FileNotFoundError:
            print(f"File {benchmark_cpp_path} not found, skipping {subfolder_name}")
            continue

        command = f".\\{subfolder_name} 1"
        output_file = os.path.join(build_path, f"{subfolder_name}_output.txt")    
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, cwd=build_path)
            compilation_time_match = re.search(r'\b\d+\.\d+\b', result.stdout)
            compilation_time = float(compilation_time_match.group()) if compilation_time_match else None
                
                # Extract depth and multiplicative depth
            depth_match = re.search(r'max:\s*\((\d+),\s*(\d+)\)', result.stdout)
            depth = int(depth_match.group(1)) if depth_match else None
            multiplicative_depth = int(depth_match.group(2)) if depth_match else None

            lines = result.stdout.splitlines()
            scalar_multiplications = None
            for i, line in enumerate(lines):
                if '|mul_plain|' in line:
                    # Check the next 1 or 2 lines for the 'total:' count
                    for j in range(1, 3):  # Check next 1 and 2 lines
                        if i + j < len(lines):
                            next_line = lines[i + j]
                            total_match = re.search(r'total:\s*(\d+)', next_line)
                            if total_match:
                                scalar_multiplications = int(total_match.group(1))
                                break
                    if scalar_multiplications is not None:
                        break

            multiplications = None
            for i, line in enumerate(lines):
                if '|mul|' in line:
                    # Check the next 1 or 2 lines for the 'total:' count
                    for j in range(1, 3):  # Check next 1 and 2 lines
                        if i + j < len(lines):
                            next_line = lines[i + j]
                            total_match = re.search(r'total:\s*(\d+)', next_line)
                            if total_match:
                                multiplications = int(total_match.group(1))
                                break
                    if multiplications is not None:
                        break

            additions = None
            for i, line in enumerate(lines):
                if '|he_add|' in line:
                    # Check the next 1 or 2 lines for the 'total:' count
                    for j in range(1, 3):  # Check next 1 and 2 lines
                        if i + j < len(lines):
                            next_line = lines[i + j]
                            total_match = re.search(r'total:\s*(\d+)', next_line)
                            if total_match:
                                additions = int(total_match.group(1))
                                break
                    if additions is not None:
                        break

            rotations = None
            for i, line in enumerate(lines):
                if '|rotate|' in line:
                    # Check the next 1 or 2 lines for the 'total:' count
                    for j in range(1, 3):  # Check next 1 and 2 lines
                        if i + j < len(lines):
                            next_line = lines[i + j]
                            total_match = re.search(r'total:\s*(\d+)', next_line)
                            if total_match:
                                rotations = int(total_match.group(1))
                                break
                    if rotations is not None:
                        break

            squares = None
            for i, line in enumerate(lines):
                if '|square|' in line:
                    # Check the next 1 or 2 lines for the 'total:' count
                    for j in range(1, 3):  # Check next 1 and 2 lines
                        if i + j < len(lines):
                            next_line = lines[i + j]
                            total_match = re.search(r'total:\s*(\d+)', next_line)
                            if total_match:
                                squares = int(total_match.group(1))
                                break
                    if squares is not None:
                        break                                  

                # Write the extracted data to the CSV file
            with open(output_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([subfolder_name, compilation_time, depth,
                                  multiplicative_depth, scalar_multiplications,
                                    multiplications,additions,rotations,squares])
            with open(output_file, 'w') as file:
                file.write(result.stdout)
                    # Step 5: Copy the content of backup.cpp to $benchmark.cpp
    
        except subprocess.CalledProcessError as e:
            print(f"Command for {subfolder_name} failed with error:\n{e.stderr}")
