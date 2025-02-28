import sys
# Define the path to the log file
def get_statistics(log_file_path):
    # Initialize an empty dictionary to store the counts
    response_counts = {}

    with open(log_file_path, 'r') as file:
        for line in file:
            if line.startswith("Response:"):
                # Extract the value between the brackets
                start_index = line.find('[') + 1
                end_index = line.find(']')
                if start_index != -1 and end_index != -1:
                    value = line[start_index:end_index]
                    # Update the count in the dictionary
                    if value in response_counts:
                        response_counts[value] += 1
                    else:
                        response_counts[value] = 1
    return response_counts


if __name__ == "__main__":
    arg = sys.argv[1]
    print(get_statistics(arg))
