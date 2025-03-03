import sys
# Define the path to the log file
def get_statistics(log_file_path):
    # Initialize an empty dictionary to store the counts and times
    response_counts = {}
    time_taken=0
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("Time taken: "):
                time_taken += float(line.replace("Time taken: ", "").strip())
                
            if line.startswith("Response:"):
                # Extract the value between the brackets
                start_index = line.find('[') + 1
                end_index = line.find(']')
                if start_index != -1 and end_index != -1:
                    value = line[start_index:end_index]
                    # Check if the next line exists and starts with "<Time_spend>: "
                    if i + 1 < len(lines) and lines[i + 1].startswith("<Time spend>:"):
                        time_spend_line = lines[i + 1]
                        # Split the time spend line by "<Time_spend>: " and get the second part
                        time_spend = time_spend_line.replace("<Time spend>:","").strip()
                        # Update the count and time in the dictionary
                        if value in response_counts:
                            response_counts[value]['count'] += 1
                            response_counts[value]['time'].append(float(time_spend))
                        else:
                            response_counts[value] = {'count': 1, 'time': [float(time_spend)]}
    for value in response_counts:
        total_time = sum(map(float, response_counts[value]['time']))
        average_time = total_time / len(response_counts[value]['time'])
        response_counts[value]['time'] = average_time
    response_counts = sorted(response_counts.items(), key=lambda x: x[1]['count'], reverse=True)
    print(f"Time taken: {time_taken}")
    return response_counts


if __name__ == "__main__":
    arg = sys.argv[1]
    print(get_statistics(arg))
