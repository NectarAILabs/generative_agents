# whisper_inner_thought_v1.py

from pydantic import BaseModel
import traceback

from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- init persona name
# !<INPUT 1>! -- whisper

template = """
Translate the following thought into a statement about !<INPUT 0>!.

Thought: "!<INPUT 1>!"
Statement: "
"""


class InnerThought(BaseModel):
  thought: str


async def run_gpt_prompt_generate_whisper_inner_thought(
  persona, whisper, test_input=None, verbose=False
):
  def create_prompt_input(persona, whisper, test_input=None):
    prompt_input = [persona.scratch.name, whisper]
    return prompt_input

  def __func_clean_up(gpt_response: InnerThought, prompt=""):
    return gpt_response.thought.split('"')[0].strip()

  def __func_validate(gpt_response, prompt=""):
    try:
      __func_clean_up(gpt_response, prompt)
      return True
    except:
      traceback.print_exc()
      return False

  def get_fail_safe():
    return "..."

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
  prompt_template = "persona/prompt_template/v2/whisper_inner_thought_v1.py"
  prompt_input = create_prompt_input(persona, whisper)
  prompt = generate_prompt(prompt_input, prompt_template_str=template)

  fail_safe = get_fail_safe()
  output = await safe_generate_structured_response(
    prompt,
    gpt_param,
    InnerThought,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
    True,
  )

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
