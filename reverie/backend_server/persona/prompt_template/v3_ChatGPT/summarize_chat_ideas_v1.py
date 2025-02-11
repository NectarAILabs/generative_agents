# summarize_chat_ideas_v1.py

import traceback

from ..common import openai_config, IdeaSummary
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response

# Variables:
# !<INPUT 0>! -- Current date
# !<INPUT 1>! -- curr_context
# !<INPUT 2>! -- currently
# !<INPUT 3>! -- Statements
# !<INPUT 4>! -- curr persona name
# !<INPUT 5>! -- target_persona.scratch.name

template = """
Current Date: !<INPUT 0>!

!<INPUT 1>!

Currently: !<INPUT 2>!

!<INPUT 3>!
Summarize the most relevant statements above that can inform !<INPUT 4>! in his conversation with !<INPUT 5>!.
"""


async def run_gpt_prompt_agent_chat_summarize_ideas(
  persona, target_persona, statements, curr_context, test_input=None, verbose=False
):
  def create_prompt_input(
    persona, target_persona, statements, curr_context, test_input=None
  ):
    prompt_input = [
      persona.scratch.get_str_curr_date_str(),
      curr_context,
      persona.scratch.currently,
      statements,
      persona.scratch.name,
      target_persona.scratch.name,
    ]
    return prompt_input

  # def __func_clean_up(gpt_response: Idea_Summary, prompt=""):
  #   return gpt_response.idea_summary

  # def __func_validate(gpt_response, prompt=""):
  #   try:
  #     __func_clean_up(gpt_response, prompt)
  #     return True
  #   except:
  #     traceback.print_exc()
  #     return False

  def get_fail_safe():
    return "..."

  # ChatGPT Plugin ===========================================================
  def __chat_func_clean_up(gpt_response: IdeaSummary, prompt=""):  ############
    return gpt_response.idea_summary

  def __chat_func_validate(gpt_response, prompt=""):  ############
    try:
      if not isinstance(gpt_response, IdeaSummary):
        return False
      return True
    except:
      traceback.print_exc()
      return False

  print("DEBUG 17")  ########
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
  prompt_input = create_prompt_input(
    persona, target_persona, statements, curr_context
  )  ########
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
  # prompt_template = "persona/prompt_template/v2/summarize_chat_ideas_v1.txt"
  # prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)
  # prompt = generate_prompt(prompt_input, prompt_template)

  # fail_safe = get_fail_safe()
  # output = safe_generate_response(prompt, gpt_param, 5, fail_safe,
  #                                  __func_validate, __func_clean_up)

  # if debug or verbose:
  #   print_run_prompts(prompt_template, persona, gpt_param,
  #                     prompt_input, prompt, output)

  # return output, [output, prompt, gpt_param, prompt_input, fail_safe]
