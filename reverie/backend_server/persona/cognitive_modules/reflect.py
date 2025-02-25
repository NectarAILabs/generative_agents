"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: reflect.py
Description: This defines the "Reflect" module for generative agents. 
"""

import datetime
# import random
# from numpy import dot
# from numpy.linalg import norm

import sys
sys.path.append('../../')
from utils import debug
from persona.prompt_template.run_gpt_prompt import (
    run_gpt_prompt_event_triple,
    run_gpt_prompt_event_poignancy,
    run_gpt_prompt_chat_poignancy,
    run_gpt_prompt_focal_pt,
    run_gpt_prompt_generate_new_schedule,
    run_gpt_prompt_insight_and_guidance,
    run_gpt_prompt_planning_thought_on_convo,
    run_gpt_prompt_memo_on_convo,
)
from persona.prompt_template.gpt_structure import get_embedding
from persona.cognitive_modules.retrieve import new_retrieve


async def generate_focal_points(persona, n=3): 
  if debug: print ("GNS FUNCTION: <generate_focal_points>")
  
  nodes = [[i.last_accessed, i]
            for i in persona.a_mem.seq_event + persona.a_mem.seq_thought
            if "idle" not in i.embedding_key]

  nodes = sorted(nodes, key=lambda x: x[0])
  nodes = [i for created, i in nodes]

  statements = ""
  for node in nodes[-1*persona.scratch.importance_ele_n:]: 
    statements += node.embedding_key + "\n"

  return (await run_gpt_prompt_focal_pt(persona, statements, n))[0]


async def generate_insights_and_evidence(persona, nodes, n=5): 
  if debug: print ("GNS FUNCTION: <generate_insights_and_evidence>")

  statements = ""
  for count, node in enumerate(nodes): 
    statements += f'{str(count)}. {node.embedding_key}\n'

  ret = (await run_gpt_prompt_insight_and_guidance(persona, statements, n))[0]

  print(ret)
  try:
    if isinstance(ret, dict):
      for thought, evi_raw in ret.items():
        evidence_node_id = [nodes[i].node_id for i in evi_raw]
        ret[thought] = evidence_node_id
      return ret
    else:
      return {"this is blank": "node_1"}
  except:
    return {"this is blank": "node_1"}


async def generate_action_event_triple(act_desp, persona): 
  """TODO 

  INPUT: 
    act_desp: the description of the action (e.g., "sleeping")
    persona: The Persona class instance
  OUTPUT: 
    a string of emoji that translates action description.
  EXAMPLE OUTPUT: 
    "🧈🍞"
  """
  if debug: print ("GNS FUNCTION: <generate_action_event_triple>")
  return (await run_gpt_prompt_event_triple(act_desp, persona))[0]


async def generate_poig_score(persona, event_type, description): 
  if debug: print ("GNS FUNCTION: <generate_poig_score>")

  if "is idle" in description: 
    return 1

  if event_type == "event" or event_type == "thought":
    response = await run_gpt_prompt_event_poignancy(persona, description)
    if response:
      return response[0]
    else:
      print(
        "ERROR: <generate_poig_score> in reflect.py: Could not get event/thought poignancy."
      )
  elif event_type == "chat":
    response = await run_gpt_prompt_chat_poignancy(
      persona, persona.scratch.act_description
    )
    if response:
      return response[0]
    else:
      print(
        "ERROR: <generate_poig_score> in reflect.py: Could not get chat poignancy."
      )


async def generate_planning_thought_on_convo(persona, all_utt):
  if debug: print ("GNS FUNCTION: <generate_planning_thought_on_convo>")
  return (await run_gpt_prompt_planning_thought_on_convo(persona, all_utt))[0]


async def generate_memo_on_convo(persona, all_utt):
  if debug: print ("GNS FUNCTION: <generate_memo_on_convo>")
  return (await run_gpt_prompt_memo_on_convo(persona, all_utt))[0]

async def generate_new_schedule_on_convo(persona, statement, start_hour):
  if debug: print ("GNS FUNCTION: <generate_new_schedule_on_convo>")
  return (await run_gpt_prompt_generate_new_schedule(persona, statement, start_hour))[0]



async def run_reflect(persona):
  """
  Run the actual reflection. We generate the focal points, retrieve any 
  relevant nodes, and generate thoughts and insights. 

  INPUT: 
    persona: Current Persona object
  Output: 
    None
  """
  # Reflection requires certain focal points. Generate that first. 
  focal_points = await generate_focal_points(persona, 3)
  # Retrieve the relevant Nodes object for each of the focal points. 
  # <retrieved> has keys of focal points, and values of the associated Nodes. 
  retrieved = await new_retrieve(persona, focal_points)

  # For each of the focal points, generate thoughts and save it in the 
  # agent's memory. 
  for focal_pt, nodes in retrieved.items(): 
    xx = [i.embedding_key for i in nodes]
    for xxx in xx: print (xxx)

    thoughts = await generate_insights_and_evidence(persona, nodes, 5)
    for thought, evidence in thoughts.items(): 
      created = persona.scratch.curr_time
      expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
      s, p, o = await generate_action_event_triple(thought, persona)  
      keywords = set([s, p, o])
      thought_poignancy = await generate_poig_score(persona, "thought", thought)
      thought_embedding_pair = (thought, await get_embedding(thought))

      persona.a_mem.add_thought(created, expiration, s, p, o, 
                                thought, keywords, thought_poignancy, 
                                thought_embedding_pair, evidence)


def reflection_trigger(persona): 
  """
  Given the current persona, determine whether the persona should run a 
  reflection. 
  
  Our current implementation checks for whether the sum of the new importance
  measure has reached the set (hyper-parameter) threshold.

  INPUT: 
    persona: Current Persona object
  Output: 
    True if we are running a new reflection. 
    False otherwise. 
  """

  if (persona.scratch.importance_trigger_curr <= 0 and 
      [] != persona.a_mem.seq_event + persona.a_mem.seq_thought): 
    return True 
  return False


def reset_reflection_counter(persona): 
  """
  We reset the counters used for the reflection trigger. 

  INPUT: 
    persona: Current Persona object
  Output: 
    None
  """
  persona_imt_max = persona.scratch.importance_trigger_max
  persona.scratch.importance_trigger_curr = persona_imt_max
  persona.scratch.importance_ele_n = 0


async def reflect(persona):
  """
  The main reflection module for the persona. We first check if the trigger 
  conditions are met, and if so, run the reflection and reset any of the 
  relevant counters. 

  INPUT: 
    persona: Current Persona object
  Output: 
    None
  """
  if reflection_trigger(persona): 
    await run_reflect(persona)
    reset_reflection_counter(persona)



  # print (persona.scratch.name, "al;sdhfjlsad", persona.scratch.chatting_end_time)
  if persona.scratch.chatting_end_time: 
    # print("DEBUG", persona.scratch.curr_time + datetime.timedelta(0,10))
    if persona.scratch.curr_time + datetime.timedelta(0,10) == persona.scratch.chatting_end_time: 
      # print ("KABOOOOOMMMMMMM")
      all_utt = ""
      if persona.scratch.chat: 
        for row in persona.scratch.chat:  
          all_utt += f"{row[0]}: {row[1]}\n"

      # planning_thought = generate_planning_thought_on_convo(persona, all_utt)
      # print ("init planning: aosdhfpaoisdh90m     ::", f"For {persona.scratch.name}'s planning: {planning_thought}")
      # planning_thought = generate_planning_thought_on_convo(target_persona, all_utt)
      # print ("target planning: aosdhfpaodish90m     ::", f"For {target_persona.scratch.name}'s planning: {planning_thought}")

      # memo_thought = generate_memo_on_convo(persona, all_utt)
      # print ("init memo: aosdhfpaoisdh90m     ::", f"For {persona.scratch.name} {memo_thought}")
      # memo_thought = generate_memo_on_convo(target_persona, all_utt)
      # print ("target memo: aosdhfpsaoish90m     ::", f"For {target_persona.scratch.name} {memo_thought}")
      

      # make sure you set the fillings as well

      # print (persona.a_mem.get_last_chat(persona.scratch.chatting_with).node_id)

      evidence = [persona.a_mem.get_last_chat(persona.scratch.chatting_with).node_id]
      planning_thought = await generate_planning_thought_on_convo(persona, all_utt)
      #Generate new schedule for the day based on the convo (added function)
      min_sum = 0 
      for i in range (persona.scratch.get_f_daily_schedule_hourly_org_index()): 
        min_sum += persona.scratch.f_daily_schedule_hourly_org[i][1]
      curr_hour = int (min_sum/60)

      if (persona.scratch.f_daily_schedule_hourly_org[persona.scratch.get_f_daily_schedule_hourly_org_index()][1] >= 120):
        start_hour = curr_hour + persona.scratch.f_daily_schedule_hourly_org[persona.scratch.get_f_daily_schedule_hourly_org_index()][1]/60

      elif (persona.scratch.f_daily_schedule_hourly_org[persona.scratch.get_f_daily_schedule_hourly_org_index()][1] + 
          persona.scratch.f_daily_schedule_hourly_org[persona.scratch.get_f_daily_schedule_hourly_org_index()+1][1]): 
        start_hour = curr_hour + ((persona.scratch.f_daily_schedule_hourly_org[persona.scratch.get_f_daily_schedule_hourly_org_index()][1] + 
                  persona.scratch.f_daily_schedule_hourly_org[persona.scratch.get_f_daily_schedule_hourly_org_index()+1][1])/60)

      else: 
        start_hour = curr_hour + 2
      start_hour = int(start_hour)
      _new_activities = await generate_new_schedule_on_convo(persona, planning_thought, start_hour)
      if len(_new_activities) >0:
        advance = persona.scratch.curr_time.replace(hour=start_hour, minute=0, second=0, microsecond=0) - persona.scratch.curr_time
        advance= int(advance.total_seconds()/60) +1
        prev_task = None
        prev_count = 0
        _new_hourly_schedule = []
        for task in _new_activities:
          if task != prev_task:
            prev_count = 1
            _new_hourly_schedule += [[task, prev_count]]
            prev_task = task
          else:
            if _new_hourly_schedule:
              _new_hourly_schedule[-1][1] += 1
        new_hourly_schedule = []
        for task, duration in _new_hourly_schedule:
          new_hourly_schedule += [[task, duration * 60]]
        
        with open("new_schedule.txt", "a") as f:
          f.write("--------------------------------\n")
          f.write(f"Persona: {persona.scratch.name}\n")
          f.write(f"Current time: {persona.scratch.curr_time}\n")
          f.write(f"New hourly schedule: {new_hourly_schedule}\n")
          f.write(f"Original hourly schedule: {persona.scratch.f_daily_schedule_hourly_org}\n")
          f.write(f"Original schedule: {persona.scratch.f_daily_schedule}\n")

          hour_index = persona.scratch.get_f_daily_schedule_hourly_org_index(advance=advance)
          persona.scratch.f_daily_schedule_hourly_org[hour_index:] = new_hourly_schedule
          schedule_index = persona.scratch.get_f_daily_schedule_index(advance=advance)
          persona.scratch.f_daily_schedule[schedule_index:] = new_hourly_schedule
          f.write(f"New hourly schedule: {persona.scratch.f_daily_schedule_hourly_org}\n")
          f.write(f"New schedule: {persona.scratch.f_daily_schedule}\n")
          f.write("--------------------------------\n")

      planning_thought = f"For {persona.scratch.name}'s planning: {planning_thought}"
      created = persona.scratch.curr_time
      expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
      s, p, o = await generate_action_event_triple(planning_thought, persona)
      keywords = set([s, p, o])
      thought_poignancy = await generate_poig_score(persona, "thought", planning_thought)
      thought_embedding_pair = (planning_thought, await get_embedding(planning_thought))

      persona.a_mem.add_thought(created, expiration, s, p, o, 
                                planning_thought, keywords, thought_poignancy, 
                                thought_embedding_pair, evidence)


      memo_thought = await generate_memo_on_convo(persona, all_utt)
      memo_thought = f"{persona.scratch.name} {memo_thought}"

      created = persona.scratch.curr_time
      expiration = persona.scratch.curr_time + datetime.timedelta(days=30)
      s, p, o = await generate_action_event_triple(memo_thought, persona)
      keywords = set([s, p, o])
      thought_poignancy = await generate_poig_score(persona, "thought", memo_thought)
      thought_embedding_pair = (memo_thought, await get_embedding(memo_thought))

      persona.a_mem.add_thought(created, expiration, s, p, o, 
                                memo_thought, keywords, thought_poignancy, 
                                thought_embedding_pair, evidence)
