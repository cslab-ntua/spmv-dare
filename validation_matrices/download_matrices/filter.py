def extract_lines_with_matrices(original_script, matrix_names):
    extracted_lines = []
    for matrix_name in matrix_names:
        with open(original_script, 'r') as file:
            for line in file:        
                if matrix_name in line:
                    extracted_lines.append(line)
                    break  # Break if a matrix name is found in the line
    return extracted_lines

def create_new_bash_script(extracted_lines, output_script):
    with open(output_script, 'w') as file:
        file.write("# build the sort-mtx executable, that will be used to sort matrix elements (by row and by column)\n")
        file.write("g++ -Wall -O3 sort-mtx.cpp -o sort-mtx\n\n")
        for line in extracted_lines:
            file.write(line)

# List of matrix names to extract
matrix_names_to_extract = [
 'spal_004', 'ldoor', 'dielFilterV2real', 'af_shell10', 'nv2', 'boneS10', 'circuit5M', 'Hook_1498', 'Geo_1438', 'Serena', 'vas_stokes_2M', 'bone010', 'audikw_1', 'Long_Coup_dt0', 'Long_Coup_dt6', 'dielFilterV3real', 'nlpkkt120', 'cage15', 'ML_Geer', 'Flan_1565', 'Cube_Coup_dt0', 'Cube_Coup_dt6', 'Bump_2911', 'vas_stokes_4M', 'nlpkkt160', 'HV15R', 'Queen_4147', 'stokes', 'nlpkkt200'
]
print('Gotta download', len(matrix_names_to_extract), 'matrices')

# Path to the original bash script
original_bash_script_path = 'SuiteSparseCollection.sh'

# Path to the new bash script to create
new_bash_script_path = 'filtered_collection.sh'

# Extract lines containing the specified matrix names
extracted_lines = extract_lines_with_matrices(original_bash_script_path, matrix_names_to_extract)

# Create a new bash script with the extracted lines
create_new_bash_script(extracted_lines, new_bash_script_path)

