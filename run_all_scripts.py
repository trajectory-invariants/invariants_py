"""
Check all examples, tests, modules and notebooks to see if they run successfully
"""

import os
import subprocess

# Directories containing the examples and tests
directories = ["examples", "tests", "invariants_py", "notebooks"]
#directories = ["notebooks"]

# Initialize an empty list to hold examples and tests that fail to execute
failed_scripts = []

# Set the matplotlib backend to Agg in the environment variable to prevent showing plots
my_env = os.environ.copy()
my_env["MPLBACKEND"] = "agg"

# Get the directory containing this script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Function to run scripts in a directory
def run_scripts(directory):
    failed_scripts_local = []
    directory_path = os.path.join(script_dir, directory)
    for root, dirs, files in os.walk(directory_path):
        # Get a list of all Python scripts in the current directory
        scripts = [f for f in files if f.endswith('.py') or f.endswith('.ipynb')]

        # Iterate over each script
        for script in scripts:
            # Construct the full script path
            script_path = os.path.join(root, script)

            # Print the name of the script being run
            print(f"Running script: {script_path}")

            # Try to execute the script
            try:
                if script.endswith('.py'):
                    subprocess.check_output(["python", script_path], env=my_env)
                elif script.endswith('.ipynb'):
                    subprocess.check_output(["jupyter", "nbconvert", "--execute", "--to", "notebook", script_path], env=my_env)
                    # Delete the generated notebook
                    os.remove(os.path.splitext(script_path)[0] + '.nbconvert.ipynb')
            except subprocess.CalledProcessError:
                # If an error occurs, add the script to the list of failed scripts
                failed_scripts_local.append(script_path)

    return failed_scripts_local

# Run scripts for each directory
for directory in directories:
    failed_scripts.extend(run_scripts(directory))

# Print out the scripts that failed to execute
print("")
print("========================================")
if failed_scripts:
    print("The following scripts failed to execute:")
    for script in failed_scripts:
        print(script)
    print("========================================")
    print("")
    raise Exception('One or more scripts failed, check the above output to see which ones.')
else:
    print("All scripts executed successfully.")
    print("========================================")
    print("")
    