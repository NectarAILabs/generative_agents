# generate_focal_pt_v1.py

import traceback

from ..common import openai_config, FocalPoint
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response
from ..v2.generate_focal_pt_v1 import run_gpt_prompt_focal_pt_v1

# Variables:
# !<INPUT 0>! -- Event/thought statements
# !<INPUT 1>! -- Count

template = """
!<INPUT 0>!

Given only the information above, what are !<INPUT 1>! most salient high-level questions we can answer about the subjects grounded in the statements?
1)
"""


async def run_gpt_prompt_focal_pt(persona, statements, n, test_input=None, verbose=False):
  def create_prompt_input(persona, statements, n, test_input=None):
    prompt_input = [statements, str(n)]
    return prompt_input

  def get_fail_safe(n):
    return ["Who am I"] * n

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response: FocalPoint, prompt=""):  ############
    ret = gpt_response.questions
    return ret

  def __chat_func_validate(gpt_response, prompt=""):  ############
    try:
      if not isinstance(gpt_response, FocalPoint):
        return False
      __chat_func_clean_up(gpt_response, prompt)
      return True
    except:
      traceback.print_exc()
      return False

  print("DEBUG 12")  ########
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
  prompt_input = create_prompt_input(persona, statements, n)  ########
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  example_output = '["What should Jane do for lunch", "Does Jane like strawberry", "Who is Jane"]'  ########
  special_instruction = "Output must be a list of str."  ########
  fail_safe = get_fail_safe(n)  ########
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    FocalPoint,
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

  # Fall back to the old version
  return run_gpt_prompt_focal_pt_v1(persona, statements, n, test_input, verbose)
