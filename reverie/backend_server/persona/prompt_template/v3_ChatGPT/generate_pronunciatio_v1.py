# generate_pronunciatio_v1.py

import re
import traceback
from pydantic import BaseModel

from ..common import openai_config
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- Action description

template = """
Convert an action description to an emoji (important: use two or less emojis).

Action description: !<INPUT 0>!
Emoji:
"""


class Pronunciatio(BaseModel):
  emoji: str


async def run_gpt_prompt_pronunciatio(action_description, persona, verbose=False):
  def create_prompt_input(action_description):
    if "(" in action_description:
      action_description = action_description.split("(")[-1].split(")")[0]
    prompt_input = [action_description]
    return prompt_input

  def get_fail_safe():
    fs = "😋"
    return fs

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response: Pronunciatio, prompt=""):  ############
    pattern = r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]"
    result = re.search(pattern, gpt_response.emoji)
    if result:
      return result.group()
    raise ValueError("No emoji found in the response.")

  def __chat_func_validate(gpt_response, prompt=""):  ############
    try:
      __chat_func_clean_up(gpt_response, prompt="")
      if len(gpt_response.emoji) == 0:
        return False
    except:
      traceback.print_exc()
      return False
    return True

  print("DEBUG 4")  ########
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
  prompt_template = (
    "persona/prompt_template/v3_ChatGPT/generate_pronunciatio_v1.py"  ########
  )
  prompt_input = create_prompt_input(action_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  example_output = "🛁🧖‍♀️"  ########
  special_instruction = (
    "The value for the output must ONLY contain the emojis."  ########
  )
  fail_safe = get_fail_safe()
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    Pronunciatio,
    example_output,
    special_instruction,
    3,
    fail_safe,
    __chat_func_validate,
    __chat_func_clean_up,
    True,
  )

  if verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

  # ChatGPT Plugin ===========================================================

  # gpt_param = {"engine": openai_config["model"], "max_tokens": 15,
  #              "temperature": 0, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop": ["\n"]}
  # prompt_template = "persona/prompt_template/v2/generate_pronunciatio_v1.txt"
  # prompt_input = create_prompt_input(action_description)

  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose:
  #   print_run_prompts(prompt_template, persona, gpt_param,
  #                     prompt_input, prompt, output)

  # return output, [output, prompt, gpt_param, prompt_input, fail_safe]
