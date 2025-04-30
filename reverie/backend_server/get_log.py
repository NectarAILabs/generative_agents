import shutil
import json
from global_methods import *

def get_log(sim_code):
  sim_storage = f"../environment/frontend_server/storage/{sim_code}"
  compressed_storage = f"../environment/frontend_server/log_agents/{sim_code}"
  persona_folder = sim_storage + "/personas"
  move_folder = sim_storage + "/movement"
  meta_file = sim_storage + "/reverie/meta.json"

  persona_names = []
  for i in find_filenames(persona_folder, ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [x]

  max_move_count = max([int(i.split("/")[-1].split(".")[0]) 
                 for i in find_filenames(move_folder, "json")])
  
  # Initialize persona_last_move dictionary for all personas
  persona_last_move = {p: {
    "movement": None,
    "pronunciatio": None,
    "description": None,
    "chat": None
  } for p in persona_names}
  
  master_move = dict()  
  for i in range(max_move_count+1): 
    master_move[i] = dict()
    with open(f"{move_folder}/{str(i)}.json") as json_file:  
      json_load = json.load(json_file)
      curr_time = json_load["meta"]["curr_time"]
      i_move_dict = json_load["persona"]
      for p in persona_names: 
        move = False
        if i == 0: 
          move = True
        elif i_move_dict[p]["movement"] != persona_last_move[p]["movement"]: 
          move = True
        coordinates = i_move_dict[p]['movement']
        pronunciatio = i_move_dict[p]['pronunciatio']
        description = i_move_dict[p]['description']
        if move and i != 0: 
          action = "moving"
          persona_last_move[p] = {"movement": i_move_dict[p]["movement"],
                                  "pronunciatio": i_move_dict[p]["pronunciatio"], 
                                  "description": i_move_dict[p]["description"], 
                                  "chat": i_move_dict[p]["chat"]}
          master_move[i][p] = {
            "timestamp": curr_time,
            "coordinates": coordinates,
            "pronunciatio": pronunciatio, 
            "description": description, 
            "action":action,
            "chat":i_move_dict[p]["chat"]}
        else:
          master_move[i][p] = {
            "timestamp": curr_time,
            "coordinates": coordinates,
            "pronunciatio": pronunciatio, 
            "description": description, 
            "action":description.split("@")[0],
            "chat":i_move_dict[p]["chat"]}


  create_folder_if_not_there(compressed_storage)
  #with open(f"{compressed_storage}/master_movement.json", "w") as outfile:
  #  outfile.write(json.dumps(master_move, indent=2))
  for p in persona_names:
    agent_log = [master_move[i][p] for i in range(len(master_move))]
    filtered_agent_log = []
    for i in range(len(agent_log)):
        if (i==len(agent_log)-1) or not (agent_log[i]['coordinates'][0] == agent_log[i-1]['coordinates'][0] and
                agent_log[i]['coordinates'][1] == agent_log[i-1]['coordinates'][1] and
                agent_log[i]['description'] == agent_log[i-1]['description'] and
                agent_log[i]['chat'] == agent_log[i-1]['chat']):
            filtered_agent_log.append(agent_log[i])
    with open(f"{compressed_storage}/{p}.json", "w") as outfile:
      outfile.write(json.dumps(filtered_agent_log, indent=2))

  shutil.copyfile(meta_file, f"{compressed_storage}/meta.json")
  shutil.copytree(persona_folder, f"{compressed_storage}/personas/")


if __name__ == '__main__':
  get_log("base_love_ville_explicit_2301_sao10k")









   











