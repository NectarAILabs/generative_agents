# poignancy_event_v1.py

import traceback
from pydantic import BaseModel

from ..common import openai_config
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response

# !<INPUT 1>!: agent name
# !<INPUT 1>!: iss
# !<INPUT 2>!: name
# !<INPUT 3>!: event description

template = """
Here is a brief description of !<INPUT 0>!. 
!<INPUT 1>!

On the scale of 1 to 10, where 1 is purely mundane (e.g., brushing teeth, making bed) and 10 is extremely poignant (e.g., a break up, college acceptance), rate the likely poignancy of the following event for !<INPUT 2>!.

Event: !<INPUT 3>!
Rate (return a number between 1 to 10):
"""


class Poignancy(BaseModel):
  poignancy: int


async def run_gpt_prompt_event_poignancy(
  persona, event_description, test_input=None, verbose=False
):
  def create_prompt_input(persona, event_description, test_input=None):
    prompt_input = [
      persona.scratch.name,
      persona.scratch.get_str_iss(),
      persona.scratch.name,
      event_description,
    ]
    return prompt_input

  # def __func_clean_up(gpt_response: Poignancy, prompt=""):
  #   response = gpt_response.poignancy
  #   return response

  # def __func_validate(gpt_response, prompt=""):
  #   try:
  #     __func_clean_up(gpt_response, prompt)
  #     return True
  #   except:
  #     traceback.print_exc()
  #     return False

  def get_fail_safe():
    return 4

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response: Poignancy, prompt=""):  ############
    response = gpt_response.poignancy
    return response

  def __chat_func_validate(gpt_response, prompt=""):  ############
    try:
      poignancy = __chat_func_clean_up(gpt_response, prompt)
      return poignancy is not None
    except:
      traceback.print_exc()
      return False

  print("DEBUG 7")  ########
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
  prompt_input = create_prompt_input(persona, event_description)  ########
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  example_output = "5"  ########
  special_instruction = "The output should ONLY contain ONE integer value on the scale of 1 to 10."  ########
  fail_safe = get_fail_safe()  ########
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    Poignancy,
    example_output,
    special_instruction,
    3,
    fail_safe,
    __chat_func_validate,
    __chat_func_clean_up,
    True,
  )
  if output != False:
    return output, [output, prompt, gpt_param, prompt_input, fail_safe]

  # ChatGPT Plugin ===========================================================

  # gpt_param = {"engine": openai_config["model"], "max_tokens": 3,
  #              "temperature": 0, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  # prompt_template = "persona/prompt_template/v2/poignancy_event_v1.txt"
  # prompt_input = create_prompt_input(persona, event_description)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose:
  #   print_run_prompts(prompt_template, persona, gpt_param,
  #                     prompt_input, prompt, output)

  # return output, [output, prompt, gpt_param, prompt_input, fail_safe]
