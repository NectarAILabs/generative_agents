import sys  
import os
import json
# Extract the logs by response type, add to module_logs folder
def extract_responses_by_type(log_file_path):
    responses_dict = {}
    
    with open(log_file_path, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if line.startswith("Response:"):
                # Extract response type from between brackets
                start_idx = line.find('[') + 1
                end_idx = line.find(']')
                if start_idx != -1 and end_idx != -1:
                    response_type = line[start_idx:end_idx]
                    
                    # Find the preceding prompt
                    prompt_start_index = None
                    for j in range(i-1, 0, -1):
                        if lines[j].startswith("Prompt:"):
                            prompt_start_index = j
                            break
                    
                    if prompt_start_index is not None:
                        # Get all lines between prompt and response
                        prompt_content = "\n".join(lines[prompt_start_index:i])
                        
                        # Add to dictionary
                        if response_type not in responses_dict:
                            responses_dict[response_type] = []
                        responses_dict[response_type].append({
                            'prompt': prompt_content,
                            'response': line.replace(f"Response: ParsedChatCompletionMessage[{response_type}]", "").strip()
                        })

    # Write to files
    log_file_name = log_file_path.split("/")[-1].replace('.txt', '')
    os.makedirs(f"./module_logs/{log_file_name}", exist_ok=True)
    
    for response_type, entries in responses_dict.items():
        output_path = f"./module_logs/{log_file_name}/{response_type}.txt"
        with open(output_path, 'w') as f:
            for entry in entries:
                f.write("=== PROMPT ===\n")
                f.write(f"{entry['prompt']}\n")
                f.write("=== RESPONSE ===\n")
                f.write(f"{entry['response']}\n")
                f.write("\n" + "="*50 + "\n\n")

    return responses_dict

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

def extract_chat_conversations(nodes_file_path):
    # Read the JSON file
    with open(nodes_file_path, 'r') as file:
        nodes = json.load(file)
    
    conversations = []
    
    # Iterate through nodes to find chat conversations
    for node in nodes:
        if node.get('predicate') == 'chat with':
            conversation = {
                'subject': node.get('subject', ''),
                'object': node.get('object', ''),
                'timestamp': node.get('created', ''),
                'content': node.get('description', '')
            }
            conversations.append(conversation)
    
    # Sort conversations by timestamp
    conversations.sort(key=lambda x: x['timestamp'])
    
    # Create output directory and file
    persona_name = nodes_file_path.split('/')[-4]  # Get persona name from path
    sim_name = nodes_file_path.split('/storage/')[-1].split('/')[0]  # Get simulation name
    
    output_dir = f"./module_logs/conversations/{sim_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Write conversations to file
    output_file = f"{output_dir}/{persona_name}_chats.txt"
    with open(output_file, 'w') as f:
        for conv in conversations:
            f.write(f"=== Conversation ===\n")
            f.write(f"Time: {conv['timestamp']}\n")
            f.write(f"Between: {conv['subject']} and {conv['object']}\n")
            f.write(f"Content: {conv['content']}\n")
            f.write("="*50 + "\n\n")
    
    return conversations

if __name__ == "__main__":
    arg = sys.argv[1]
    extract_responses_by_type(arg)
    print(get_statistics(arg))
    #nodes_file_path = sys.argv[2]
    #conversations = extract_chat_conversations(nodes_file_path)
    #print(f"Extracted {len(conversations)} conversations")
