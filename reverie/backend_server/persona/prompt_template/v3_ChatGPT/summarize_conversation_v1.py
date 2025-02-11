# summarize_conversation_v1.py

import traceback
from pydantic import BaseModel

from ..common import openai_config
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response

# Variables:
# !<INPUT 0>! -- init_persona_name


template = """
Conversation: 
!<INPUT 0>!

Summarize the conversation above in one sentence:
This is a conversation about
"""


class ConversationSummary(BaseModel):
  summary: str


async def run_gpt_prompt_summarize_conversation(
  persona, conversation, test_input=None, verbose=False
):
  def create_prompt_input(conversation, test_input=None):
    convo_str = ""
    for row in conversation:
      convo_str += f'{row[0]}: "{row[1]}"\n'

    prompt_input = [convo_str]
    return prompt_input

  # def __func_clean_up(gpt_response: ConversationSummary, prompt=""):
  #   ret = "conversing about " + gpt_response.conversation.strip()
  #   return ret

  # def __func_validate(gpt_response, prompt=""):
  #   try:
  #     __func_clean_up(gpt_response, prompt)
  #     return True
  #   except:
  #     traceback.print_exc()
  #     return False

  def get_fail_safe():
    return "conversing with a housemate about morning greetings"

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response: ConversationSummary, prompt=""):  ############
    ret = "conversing about " + gpt_response.summary.strip()
    return ret

  def __chat_func_validate(gpt_response, prompt=""):  ############
    try:
      __chat_func_clean_up(gpt_response, prompt)
      return True
    except:
      traceback.print_exc()
      return False

  print("DEBUG 11")  ########
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
  prompt_input = create_prompt_input(conversation, test_input)  ########
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  example_output = "conversing about what to eat for lunch"  ########
  special_instruction = "The output must continue the sentence above by filling in the <fill in> tag. Don't start with 'this is a conversation about...' Just finish the sentence but do not miss any important details (including who are chatting)."  ########
  fail_safe = get_fail_safe()  ########
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    ConversationSummary,
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

  # gpt_param = {"engine": openai_config["model"], "max_tokens": 50,
  #              "temperature": 0, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  # prompt_template = "persona/prompt_template/v2/summarize_conversation_v1.txt"
  # prompt_input = create_prompt_input(conversation, test_input)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose:
  #   print_run_prompts(prompt_template, persona, gpt_param,
  #                     prompt_input, prompt, output)

  # return output, [output, prompt, gpt_param, prompt_input, fail_safe]
