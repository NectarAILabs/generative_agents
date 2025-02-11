# summarize_ideas_v1.py

import traceback

from ..common import openai_config, IdeaSummary
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response

# Variables:
# !<INPUT 0>! -- Statements
# !<INPUT 1>! -- agent name
# !<INPUT 2>! -- interviewer question

template = """
Statements: 
!<INPUT 0>!

An interviewer said to !<INPUT 1>!: 
"!<INPUT 2>!"

Summarize the Statements that are most relevant to the interviewer's line:
"""


async def run_gpt_prompt_summarize_ideas(
  persona, statements, question, test_input=None, verbose=False
):
  def create_prompt_input(persona, statements, question, test_input=None):
    prompt_input = [statements, persona.scratch.name, question]
    return prompt_input

  # def __func_clean_up(gpt_response: Idea_Summary, prompt=""):
  #   return gpt_response.idea_summary.strip()

  # def __func_validate(gpt_response, prompt=""):
  #   try:
  #     gpt_response = __func_clean_up(gpt_response, prompt)
  #     if gpt_response is None:
  #       return False
  #   except:
  #     traceback.print_exc()
  #     return False
  #   return True

  def get_fail_safe():
    return "..."

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response: IdeaSummary, prompt=""):  ############
    return gpt_response.idea_summary.strip()

  def __chat_func_validate(gpt_response, prompt=""):  ############
    try:
      response = __chat_func_clean_up(gpt_response, prompt)
      if response is None or response == "":
        return False
      return True
    except:
      traceback.print_exc()
      return False

  print("DEBUG 16")  ########
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
  prompt_input = create_prompt_input(persona, statements, question)  ########
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  example_output = "Jane Doe is working on a project"  ########
  special_instruction = (
    "The output should be a string that responds to the question."  ########
  )
  fail_safe = get_fail_safe()  ########
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    IdeaSummary,
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

  # gpt_param = {"engine": openai_config["model"], "max_tokens": 150,
  #              "temperature": 0.5, "top_p": 1, "stream": False,
  #              "frequency_penalty": 0, "presence_penalty": 0, "stop": None}
  # prompt_template = "persona/prompt_template/v2/summarize_ideas_v1.txt"
  # prompt_input = create_prompt_input(persona, statements, question)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose:
  #   print_run_prompts(prompt_template, persona, gpt_param,
  #                     prompt_input, prompt, output)

  # return output, [output, prompt, gpt_param, prompt_input, fail_safe]
