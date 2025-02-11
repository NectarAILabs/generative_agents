# Variables:
# !<INPUT 0>! -- Comment

import traceback
from pydantic import BaseModel
import asyncio
from ..common import openai_config
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response


template = """
The following line was submitted to a chatbot by a user. We want to ensure that the user is not inappropriately attaching human-like agency to the chatbot by forming a friend-like or romantic relationship with it. Does the user's line cross the line and raise concerns? Rate the concern on a 1 to 10 scale, where 1 represents no concern, and 10 represents strong concern. 

Comment: "!<INPUT 0>!"
--
Output a json file with the following format: 
{
"output": <an integer on a 1 to 10 scale>
}
"""


class SafetyScore(BaseModel):
  # Safety score should range 1-10
  safety_score: int


async def run_gpt_generate_safety_score(persona, comment, test_input=None, verbose=False):
  def create_prompt_input(comment, test_input=None):
    prompt_input = [comment]
    return prompt_input

  def __chat_func_clean_up(gpt_response: SafetyScore, prompt=""):
    score = gpt_response.safety_score
    if isinstance(score, int) and 1 <= score <= 10:
      return score
    raise ValueError("Output is not a valid integer between 1 and 10")

  def __chat_func_validate(gpt_response, prompt=""):
    try:
      __chat_func_clean_up(gpt_response)
    except:
      traceback.print_exc()
      return False

  def get_fail_safe():
    return None

  print("11")
  prompt_input = create_prompt_input(comment)
  print("22")
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  print(prompt)
  fail_safe = get_fail_safe()
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    SafetyScore,
    repeat=3,
    fail_safe_response=fail_safe,
    func_validate=__chat_func_validate,
    func_clean_up=__chat_func_clean_up,
    verbose=verbose,
  )
  print(output)

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
  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
