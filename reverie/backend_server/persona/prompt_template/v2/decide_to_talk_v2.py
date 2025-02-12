# decide_to_talk_v1.py

from pydantic import BaseModel
import traceback

from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

template = """
Task -- given context, determine whether the subject will initiate a conversation with another.
Format:
Context: []
Question: []
Reasoning: []
Answer in "yes" or "no": []
---
Context: !<INPUT 0>!
Right now, it is !<INPUT 1>!. !<INPUT 2>! and !<INPUT 3>! last chatted at !<INPUT 4>! about !<INPUT 5>!.
!<INPUT 6>!
!<INPUT 7>!

Question: Would !<INPUT 8>! initiate a conversation with !<INPUT 9>!?

Reasoning: Let's think step by step.
"""


class DecideToTalk(BaseModel):
  decision: bool


async def run_gpt_prompt_decide_to_talk(
  persona, target_persona, retrieved, test_input=None, verbose=False
):
  def create_prompt_input(init_persona, target_persona, retrieved, test_input=None):
    last_chat = init_persona.a_mem.get_last_chat(target_persona.name)
    last_chatted_time = ""
    last_chat_about = ""
    if last_chat:
      last_chatted_time = last_chat.created.strftime("%B %d, %Y, %H:%M:%S")
      last_chat_about = last_chat.description

    context = ""
    for c_node in retrieved["events"]:
      curr_desc = c_node.description.split(" ")
      curr_desc[2:3] = ["was"]
      curr_desc = " ".join(curr_desc)
      context += f"{curr_desc}. "
    context += "\n"
    for c_node in retrieved["thoughts"]:
      context += f"{c_node.description}. "

    curr_time = init_persona.scratch.curr_time.strftime("%B %d, %Y, %H:%M:%S %p")
    init_act_desc = init_persona.scratch.act_description
    if "(" in init_act_desc:
      init_act_desc = init_act_desc.split("(")[-1][:-1]

    if len(init_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
      init_p_desc = f"{init_persona.name} is already {init_act_desc}"
    elif "waiting" in init_act_desc:
      init_p_desc = f"{init_persona.name} is {init_act_desc}"
    else:
      init_p_desc = f"{init_persona.name} is on the way to {init_act_desc}"

    target_act_desc = target_persona.scratch.act_description
    if "(" in target_act_desc:
      target_act_desc = target_act_desc.split("(")[-1][:-1]

    if len(target_persona.scratch.planned_path) == 0 and "waiting" not in init_act_desc:
      target_p_desc = f"{target_persona.name} is already {target_act_desc}"
    elif "waiting" in init_act_desc:
      target_p_desc = f"{init_persona.name} is {init_act_desc}"
    else:
      target_p_desc = f"{target_persona.name} is on the way to {target_act_desc}"

    prompt_input = []
    prompt_input += [context]

    prompt_input += [curr_time]

    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    prompt_input += [last_chatted_time]
    prompt_input += [last_chat_about]

    prompt_input += [init_p_desc]
    prompt_input += [target_p_desc]
    prompt_input += [init_persona.name]
    prompt_input += [target_persona.name]
    return prompt_input

  def __func_clean_up(gpt_response: DecideToTalk, prompt=""):
    return "yes" if gpt_response.decision is True else "no"

  def __func_validate(gpt_response, prompt=""):
    try:
      if isinstance(gpt_response, DecideToTalk) and __func_clean_up(
        gpt_response, prompt
      ) in ["yes", "no"]:
        return True
      return False
    except:
      traceback.print_exc()
      return False

  def get_fail_safe():
    fs = "yes"
    return fs

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
  prompt_template = "persona/prompt_template/v2/decide_to_talk_v2.py"
  prompt_input = create_prompt_input(persona, target_persona, retrieved, test_input)
  prompt = generate_prompt(prompt_input, prompt_template_str=template)

  fail_safe = get_fail_safe()
  output = await safe_generate_structured_response(
    prompt, gpt_param, DecideToTalk, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
