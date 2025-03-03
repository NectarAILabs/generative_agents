from pydantic import BaseModel
import traceback
from typing import Any, Optional
import datetime
from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts

#Modified to have more specific information for the replan.
def create_prompt(prompt_input: dict[str, Any]):
  conversation = prompt_input["conversation"]
  persona_1_name = prompt_input["persona_1_name"]
  persona_2_name = prompt_input["persona_2_name"]
  curr_time = prompt_input["curr_time"]
  sector_accessibles_str = prompt_input["sector_accessibles_str"]
  retrieved_str = prompt_input["retrieved_str"]
  prompt = f"""
[Conversation]
{conversation}
[End of conversation]

[Context]
{retrieved_str}

Current time now is {curr_time.strftime('%B %d, %Y %I:%M %p')}
Write down if there is anything from the conversation that both personas needs to remember for their meeting, in a full sentence, under 100 words.
Remember to generate the date, time and also the sector (where they both know and can go) of the meeting (persona's house, restaurant, pub, coffee shop,...), as well as both persona's names.
Planning date should be in form 'YYYY-MM-DD HH:MM AM/PM' and should be as soon as possible. If they don't mention about it, just assume it's today.
If there's nothing to remember, the planning thought should be empty and you can assign any planning date.
ALL sectors that they both can go are:
{sector_accessibles_str}
"""
  return prompt


class PlanningThought(BaseModel):
  planning_thought: str
  planning_date: str
  planning_sector: str

async def run_gpt_prompt_planning_thought_on_convo(
  persona, target_persona, all_utterances, maze, personas, retrieved, verbose=False
):
  def create_prompt_input(persona,target_persona, all_utterances, maze, personas, retrieved, test_input=None):
    init_persona_world = f"{maze.access_tile(persona.scratch.curr_tile)['world']}"
    init_persona_sector_accessibles = [i.strip() for i in persona.s_mem.get_str_accessible_sectors(init_persona_world).split(",")]
    target_persona_world = f"{maze.access_tile(target_persona.scratch.curr_tile)['world']}"
    target_persona_sector_accessibles = [i.strip() for i in target_persona.s_mem.get_str_accessible_sectors(target_persona_world).split(",")]
    sector_accessibles = list(set(init_persona_sector_accessibles + target_persona_sector_accessibles))
    sector_accessibles_str = ", ".join(sector_accessibles)
    set_retrieved = set()
    retrieved_str = ""
    for key, vals in retrieved.items():
      for v in vals:
        if v not in set_retrieved:
          set_retrieved.add(v)
          retrieved_str += f"- {v.description}\n"
    prompt_input = {
      "conversation": all_utterances,
      "persona_1_name": persona.scratch.name,
      "persona_2_name": target_persona.scratch.name,
      "curr_time": persona.scratch.curr_time,
      "sector_accessibles_str": sector_accessibles_str,
      "retrieved_str": retrieved_str
    }
    return prompt_input

  def __func_clean_up(gpt_response: PlanningThought, prompt=""):
    if gpt_response.planning_thought == "":
      return ""
    if gpt_response.planning_date is None:
      return ""
    return gpt_response.planning_thought.strip().strip('".').strip() + f" at {gpt_response.planning_date} at {gpt_response.planning_sector}"

  def __func_validate(gpt_response, prompt=""):
    try:
      if not isinstance(gpt_response, PlanningThought):
        return False
      __func_clean_up(gpt_response, prompt)
      return True
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    return ""

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 300,
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(persona, target_persona, all_utterances, maze, personas, retrieved)
  prompt = create_prompt(prompt_input)
  provider_parameter = openai_config.get("other_providers", {}).get("planning_thought_provider", None)
  if provider_parameter != None:
    gpt_param.update({k:v for k,v in provider_parameter.items() if k != "model"})
    gpt_param["engine"] = provider_parameter["model"]
  fail_safe = get_fail_safe()
  output = await safe_generate_structured_response(
    prompt,
    gpt_param,
    PlanningThought,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
