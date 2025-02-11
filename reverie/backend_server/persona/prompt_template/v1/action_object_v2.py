from pydantic import BaseModel
import random
import asyncio
from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- curr action seq
# !<INPUT 1>! -- Objects available

template = """
Current activity: sleep in bed
Objects available: {bed, easel, closet, painting}
Pick ONE most relevant object from the objects available: bed
---
Current activity: painting
Objects available: {easel, closet, sink, microwave}
Pick ONE most relevant object from the objects available: easel
---
Current activity: cooking
Objects available: {stove, sink, fridge, counter}
Pick ONE most relevant object from the objects available: stove
---
Current activity: watch TV
Objects available: {couch, TV, remote, coffee table}
Pick ONE most relevant object from the objects available: TV
---
Current activity: study
Objects available: {desk, computer, chair, bookshelf}
Pick ONE most relevant object from the objects available: desk
---
Current activity: talk on the phone
Objects available: {phone, charger, bed, nightstand}
Pick ONE most relevant object from the objects available: phone
---
Current activity: !<INPUT 0>!
Objects available: {!<INPUT 1>!}
Pick ONE most relevant object from the objects available:
"""


class GameObject(BaseModel):
  object: str


async def run_gpt_prompt_action_game_object(
  action_description, persona, maze, temp_address, test_input=None, verbose=False
):
  def create_prompt_input(action_description, persona, temp_address, test_input=None):
    prompt_input = []
    if "(" in action_description:
      action_description = action_description.split("(")[-1][:-1]

    prompt_input += [action_description]
    prompt_input += [persona.s_mem.get_str_accessible_arena_game_objects(temp_address)]
    return prompt_input

  def __func_validate(gpt_response, prompt=""):
    if len(gpt_response.object.strip()) < 1:
      return False
    return True

  def __func_clean_up(gpt_response: GameObject, prompt=""):
    return gpt_response.object

  def get_fail_safe():
    fs = "<random>"
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
  prompt_template = "persona/prompt_template/v1/action_object_v2.py"
  prompt_input = create_prompt_input(
    action_description, persona, temp_address, test_input
  )
  prompt = generate_prompt(prompt_input, prompt_template_str=template)

  fail_safe = get_fail_safe()
  output = await safe_generate_structured_response(
    prompt, gpt_param, GameObject, 5, fail_safe, __func_validate, __func_clean_up
  )

  x = [
    i.strip()
    for i in persona.s_mem.get_str_accessible_arena_game_objects(temp_address).split(
      ","
    )
  ]
  if output not in x:
    print("ERROR: Output is not an accessible game object:", output)
    print("Choosing a random accessible object instead.")
    output = random.choice(x)
    print("Randomly chosen object:", output)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
