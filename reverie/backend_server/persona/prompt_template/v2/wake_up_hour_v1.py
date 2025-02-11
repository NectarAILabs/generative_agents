# wake_up_hour_v1.py

from pydantic import BaseModel
import traceback

from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- Identity Stable Set
# !<INPUT 1>! -- Lifestyle
# !<INPUT 2>! -- Persona first names

template = """
!<INPUT 0>!

In general, !<INPUT 1>!
!<INPUT 2>!'s wake up hour:
"""


class WakeUpHour(BaseModel):
  wake_up_hour: int


async def run_gpt_prompt_wake_up_hour(persona, test_input=None, verbose=False):
  """
  Given the persona, returns an integer that indicates the hour when the
  persona wakes up.

  INPUT:
    persona: The Persona class instance
  OUTPUT:
    integer for the wake up hour.
  """

  def create_prompt_input(persona, test_input=None):
    if test_input:
      return test_input
    prompt_input = [
      persona.scratch.get_str_iss(),
      persona.scratch.get_str_lifestyle(),
      persona.scratch.get_str_firstname(),
    ]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    if isinstance(gpt_response.wake_up_hour, int):
      return gpt_response.wake_up_hour
    raise TypeError("wake_up_hour of type int not found")

  def __func_validate(gpt_response, prompt=""):
    try:
      __func_clean_up(gpt_response, prompt="")
    except:
      traceback.print_exc()
      return False
    return True

  def get_fail_safe():
    fs = 8
    return fs

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 100,
    "temperature": 0.8,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": ["\n"],
  }
  prompt_template = "persona/prompt_template/v2/wake_up_hour_v1.py"
  prompt_input = create_prompt_input(persona, test_input)
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  fail_safe = get_fail_safe()

  output = await safe_generate_structured_response(
    prompt, gpt_param, WakeUpHour, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
