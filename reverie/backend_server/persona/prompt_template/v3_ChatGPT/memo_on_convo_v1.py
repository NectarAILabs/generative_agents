# memo_on_convo_v1.py

import traceback

from ..common import openai_config, ConvoTakeaways
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response
from ..v2.memo_on_convo_v1 import run_gpt_prompt_memo_on_convo_v1

# Variables:
# !<INPUT 0>! -- All convo utterances
# !<INPUT 1>! -- persona name
# !<INPUT 2>! -- persona name
# !<INPUT 3>! -- persona name

template = """
[Conversation]
!<INPUT 0>!

Write down if there is anything from the conversation that !<INPUT 1>! might have found interesting from !<INPUT 2>!'s perspective, in a full sentence.

"!<INPUT 3>!
"""


async def run_gpt_prompt_memo_on_convo(persona, all_utt, test_input=None, verbose=False):
  def create_prompt_input(persona, all_utt, test_input=None):
    prompt_input = [
      all_utt,
      persona.scratch.name,
      persona.scratch.name,
      persona.scratch.name,
    ]
    return prompt_input

  def get_fail_safe():
    return "..."

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response: ConvoTakeaways, prompt=""):  ############
    return gpt_response.takeaway

  def __chat_func_validate(gpt_response, prompt=""):  ############
    try:
      if not isinstance(gpt_response, ConvoTakeaways):
        return False
      __chat_func_clean_up(gpt_response, prompt)
      return True
    except:
      traceback.print_exc()
      return False

  print("DEBUG 15")  ########
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
  prompt_input = create_prompt_input(persona, all_utt)  ########
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  example_output = "Jane Doe was interesting to talk to."  ########
  special_instruction = "The output should ONLY contain a string that summarizes anything interesting that the agent may have noticed"  ########
  fail_safe = get_fail_safe()  ########
  output = ChatGPT_safe_generate_structured_response(
    prompt,
    ConvoTakeaways,
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
  return run_gpt_prompt_memo_on_convo_v1(persona, all_utt, test_input, verbose)
