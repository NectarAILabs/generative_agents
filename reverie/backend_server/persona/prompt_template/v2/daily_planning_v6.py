# daily_planning_v6.py

from pydantic import BaseModel
import traceback

from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- Commonset
# !<INPUT 1>! -- Lifestyle
# !<INPUT 2>! -- Reverie date time now
# !<INPUT 3>! -- Persona first names
# !<INPUT 4>! -- wake_up_hour

template = """
!<INPUT 0>!

In general, !<INPUT 1>!
Today is !<INPUT 2>!. Describe !<INPUT 3>!'s plan for the whole day, from morning 'til night, in broad-strokes. Include the time of the day. e.g., "wake up and complete their morning routine at !<INPUT 4>!", "have lunch at 12:00 pm", "watch TV from 7 to 8 pm".
"""


class DailyPlan(BaseModel):
  daily_plan: list[str]


async def run_gpt_prompt_daily_plan(persona, wake_up_hour, test_input=None, verbose=False):
  """
  Basically the long term planning that spans a day. Returns a list of actions
  that the persona will take today. Usually comes in the following form:
  'wake up and complete the morning routine at 6:00 am',
  'eat breakfast at 7:00 am',..
  Note that the actions come without a period.

  INPUT:
    persona: The Persona class instance
  OUTPUT:
    a list of daily actions in broad strokes.
  """

  def create_prompt_input(persona, wake_up_hour, test_input=None):
    if test_input:
      return test_input
    prompt_input = []
    prompt_input += [persona.scratch.get_str_iss()]
    prompt_input += [persona.scratch.get_str_lifestyle()]
    prompt_input += [persona.scratch.get_str_curr_date_str()]
    prompt_input += [persona.scratch.get_str_firstname()]
    prompt_input += [f"{str(wake_up_hour)}:00 am"]
    return prompt_input

  def __func_clean_up(gpt_response, prompt=""):
    return gpt_response.daily_plan

  def __func_validate(gpt_response, prompt=""):
    try:
      __func_clean_up(gpt_response, prompt="")
    except:
      traceback.print_exc()
      return False
    return True

  def get_fail_safe():
    fs = [
      "wake up and complete the morning routine at 6:00 am",
      "eat breakfast at 7:00 am",
      "read a book from 8:00 am to 12:00 pm",
      "have lunch at 12:00 pm",
      "take a nap from 1:00 pm to 4:00 pm",
      "relax and watch TV from 7:00 pm to 8:00 pm",
      "go to bed at 11:00 pm",
    ]
    return fs

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 2000,
    "temperature": 1,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_template = "persona/prompt_template/v2/daily_planning_v6.py"
  prompt_input = create_prompt_input(persona, wake_up_hour, test_input)
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  fail_safe = get_fail_safe()

  output = await safe_generate_structured_response(
    prompt, gpt_param, DailyPlan, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
