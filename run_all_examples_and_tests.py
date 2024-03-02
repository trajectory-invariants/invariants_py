import os
import subprocess

# Directories containing the examples and tests
directories = ["examples", "tests"]

# Initialize an empty list to hold examples and tests that fail to execute
failed_scripts = []

# Function to run scripts in a directory
def run_scripts(directory):
    failed_scripts_local = []
    for root, dirs, files in os.walk(directory):
        # Get a list of all Python scripts in the current directory
        scripts = [f for f in files if f.endswith('.py')]

        # Iterate over each script
        for script in scripts:
            # Construct the full script path
            script_path = os.path.join(root, script)

            # Print the name of the script being run
            print(f"Running script: {script_path}")

            # Try to execute the script
            try:
                subprocess.check_output(["python", script_path])
            except subprocess.CalledProcessError:
                # If an error occurs, add the script to the list of failed scripts
                failed_scripts_local.append(script_path)

    return failed_scripts_local

# Run scripts for each directory
for directory in directories:
    failed_scripts.extend(run_scripts(directory))

# Print out the scripts that failed to execute
if failed_scripts:
    print("The following scripts failed to execute:")
    for script in failed_scripts:
        print(script)
    raise Exception('One or more scripts failed, see the above output')
else:
    print("All scripts executed successfully.")
