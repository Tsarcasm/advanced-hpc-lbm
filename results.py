import os
import re
import statistics

def handle_contents(contents):
    # Use a regular expression to extract the elapsed total time from the log file
    pattern = r"Elapsed Total time:\s+(\d+\.\d+) \(s\)"
    match = re.search(pattern, contents)
    if match is not None:
        elapsed_time = float(match.group(1))
        return elapsed_time
    else:
        raise ValueError("Could not extract elapsed time from log file")

output_dir = "output"  # Change this to match your output directory name
output_prefix = "d2q9-bgk_"

elapsed_times = []  # List to store elapsed times from each log file

# Iterate over all files in the output directory that have the specified prefix
for filename in os.listdir(output_dir):
    if filename.startswith(output_prefix):
        output_file = os.path.join(output_dir, filename)
        with open(output_file, "r") as f:
            contents = f.read()
            elapsed_time = handle_contents(contents)
            elapsed_times.append(elapsed_time)

# Print some statistics on the elapsed times
if len(elapsed_times) > 0:
    min_time = min(elapsed_times)
    max_time = max(elapsed_times)
    mean_time = sum(elapsed_times) / len(elapsed_times)
    variance = statistics.variance(elapsed_times)


    print("Minimum time (s): {:.2f}".format(min_time))
    print("Maximum time (s): {:.2f}".format(max_time))
    print("Mean time (s): {:.2f}".format(mean_time))
    print("Variance (s^2): {:.4f}".format(variance))
else:
    print("No log files found")
