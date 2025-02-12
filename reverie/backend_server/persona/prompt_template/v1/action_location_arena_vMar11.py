from utils import debug
from ..common import ActionLoc, openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts
import asyncio
# Variables:
# !<INPUT 0>! -- Persona name
# !<INPUT 1>! -- Persona's current arena
# !<INPUT 2>! -- Persona's current sector
# !<INPUT 3>! -- Persona name
# !<INPUT 4>! -- target sector
# !<INPUT 5>! -- Persona's sector's all arenas (minus no access)
# !<INPUT 6>! -- Curr action seq
# !<INPUT 7>! -- Persona name
# !<INPUT 8>! -- Persona's current sector

template = """
Jane Anderson is in kitchen in Jane Anderson's house.
Jane Anderson is going to Jane Anderson's house that has the following areas: {kitchen, bedroom, bathroom}
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For cooking, Jane Anderson should go to the following area in Jane Anderson's house:
Answer: {kitchen}
---
Tom Watson is in common room in Tom Watson's apartment.
Tom Watson is going to Hobbs Cafe that has the following areas: {cafe}
Stay in the current area if the activity can be done there. Never go into other people's rooms unless necessary.
For getting coffee, Tom Watson should go to the following area in Hobbs Cafe:
Answer: {cafe}
---

!<INPUT 0>! is going to !<INPUT 1>! that has the following areas: {!<INPUT 2>!}
* Stay in the current area if the activity can be done there.
* NEVER go into other people's rooms unless necessary.
!<INPUT 3>! is !<INPUT 4>!. For !<INPUT 5>!, !<INPUT 6>! should go to the following area in !<INPUT 7>! (MUST pick one of {!<INPUT 8>!}):
Answer: {
"""


async def run_gpt_prompt_action_arena(
  action_description,
  persona,
  maze,
  act_world,
  act_sector,
  test_input=None,
  verbose=False,
):
  def create_prompt_input(
    action_description, persona, maze, act_world, act_sector, test_input=None
  ):
    prompt_input = []
    # prompt_input += [persona.scratch.get_str_name()]
    # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
    # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["sector"]]
    prompt_input += [persona.scratch.get_str_name()]
    x = f"{act_world}:{act_sector}"
    prompt_input += [act_sector]

    # MAR 11 TEMP
    accessible_arena_str = persona.s_mem.get_str_accessible_sector_arenas(x)
    curr = accessible_arena_str.split(", ")
    fin_accessible_arenas = []
    for i in curr:
      if "'s room" in i:
        if persona.scratch.last_name in i:
          fin_accessible_arenas += [i]
      else:
        fin_accessible_arenas += [i]
    accessible_arena_str = ", ".join(fin_accessible_arenas)
    # END MAR 11 TEMP

    prompt_input += [accessible_arena_str]

    action_description_1 = action_description
    action_description_2 = action_description
    if "(" in action_description:
      action_description_1 = action_description.split("(")[0].strip()
      action_description_2 = action_description.split("(")[-1][:-1]
    prompt_input += [persona.scratch.get_str_name()]
    prompt_input += [action_description_1]

    prompt_input += [action_description_2]
    prompt_input += [persona.scratch.get_str_name()]

    prompt_input += [act_sector]

    prompt_input += [accessible_arena_str]
    # prompt_input += [maze.access_tile(persona.scratch.curr_tile)["arena"]]
    # x = f"{maze.access_tile(persona.scratch.curr_tile)['world']}:{maze.access_tile(persona.scratch.curr_tile)['sector']}:{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    # prompt_input += [persona.s_mem.get_str_accessible_arena_game_objects(x)]

    return prompt_input

  def __func_clean_up(gpt_response: ActionLoc, prompt=""):
    return gpt_response.area

  def __func_validate(gpt_response, prompt=""):
    arena = __func_clean_up(gpt_response)
    if len(arena.strip()) < 1:
      return False
    if "}" in arena:
      return False
    if "," in arena:
      return False
    return True

  def get_fail_safe():
    fs = "main room"
    return fs

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 100,
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_template = "persona/prompt_template/v1/action_location_object_vMar11.py"
  prompt_input = create_prompt_input(
    action_description, persona, maze, act_world, act_sector
  )
  prompt = generate_prompt(prompt_input, prompt_template_str=template)

  fail_safe = get_fail_safe()
  output = await safe_generate_structured_response(
    prompt,
    gpt_param,
    ActionLoc,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
    verbose=False,
  )

  # y = f"{act_world}:{act_sector}"
  # x = [i.strip() for i in persona.s_mem.get_str_accessible_sector_arenas(y).split(",")]
  # if output not in x:
  #   output = random.choice(x)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
