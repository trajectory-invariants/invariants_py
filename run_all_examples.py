import os
import subprocess

# Directory containing the examples
dir_name = "examples"

# Initialize an empty list to hold examples that fail to execute
failed_examples = []

# Use os.walk to traverse all subdirectories
for root, dirs, files in os.walk(dir_name):
    # Get a list of all Python examples in the current directory
    examples = [f for f in files if f.endswith('.py')]

    # Iterate over each example
    for example in examples:
        # Construct the full example path
        example_path = os.path.join(root, example)
        
        # Print the name of the example being run
        print(f"Running example: {example_path}")
       
        # Try to execute the example
        try:
            subprocess.check_output(["python", example_path])
        except subprocess.CalledProcessError:
            # If an error occurs, add the example to the list of failed examples
            failed_examples.append(example_path)

# Print out the examples that failed to execute
if failed_examples:
    print("The following examples failed to execute:")
    for example in failed_examples:
        print(example)
    raise Exception('One or more examples failed, see the above output')
else:
    print("All examples executed successfully.")