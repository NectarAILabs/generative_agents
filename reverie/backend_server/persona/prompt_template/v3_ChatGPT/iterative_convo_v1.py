# iterative_convo_v1.py

import traceback
from pydantic import BaseModel

from ..common import openai_config
from ..gpt_structure import generate_prompt, ChatGPT_safe_generate_structured_response

# Variables:
# !<INPUT 0>! -- persona ISS
# !<INPUT 1>! -- persona name
# !<INPUT 2>! -- retrieved memory
# !<INPUT 3>! -- past context
# !<INPUT 4>! -- current location
# !<INPUT 5>! -- current context
# !<INPUT 6>! -- persona name
# !<INPUT 7>! -- target persona name
# !<INPUT 8>! -- curr convo
# !<INPUT 9>! -- persona name
# !<INPUT 10>! -- target persona name


template = """
Context for the task:

PART 1.
!<INPUT 0>!

Here is the memory that is in !<INPUT 1>!'s head:
!<INPUT 2>!

PART 2.
Past Context:
!<INPUT 3>!

Current Location: !<INPUT 4>!

Current Context:
!<INPUT 5>!

!<INPUT 6>! and !<INPUT 7>! are chatting. Here is their conversation so far:
!<INPUT 8>!

---
Task: Given the above, what should !<INPUT 9>! say to !<INPUT 10>! next in the conversation? And did it end the conversation?
"""


class ChatUtterance(BaseModel):
  utterance: str
  did_conversation_end: bool


async def run_gpt_generate_iterative_chat_utt(
  maze,
  init_persona,
  target_persona,
  retrieved,
  curr_context,
  curr_chat,
  test_input=None,
  verbose=False,
):
  def create_prompt_input(
    maze,
    init_persona,
    target_persona,
    retrieved,
    curr_context,
    curr_chat,
    test_input=None,
  ):
    persona = init_persona
    prev_convo_insert = "\n"
    if persona.a_mem.seq_chat:
      for i in persona.a_mem.seq_chat:
        if i.object == target_persona.scratch.name:
          v1 = int((persona.scratch.curr_time - i.created).total_seconds() / 60)
          prev_convo_insert += f"{str(v1)} minutes ago, {persona.scratch.name} and {target_persona.scratch.name} were already {i.description} This context takes place after that conversation."
          break
    if prev_convo_insert == "\n":
      prev_convo_insert = ""
    if persona.a_mem.seq_chat:
      if (
        int(
          (
            persona.scratch.curr_time - persona.a_mem.seq_chat[-1].created
          ).total_seconds()
          / 60
        )
        > 480
      ):
        prev_convo_insert = ""
    print(prev_convo_insert)

    curr_sector = f"{maze.access_tile(persona.scratch.curr_tile)['sector']}"
    curr_arena = f"{maze.access_tile(persona.scratch.curr_tile)['arena']}"
    curr_location = f"{curr_arena} in {curr_sector}"

    retrieved_str = ""
    for key, vals in retrieved.items():
      for v in vals:
        retrieved_str += f"- {v.description}\n"

    convo_str = ""
    for i in curr_chat:
      convo_str += ": ".join(i) + "\n"
    if convo_str == "":
      convo_str = "[The conversation has not started yet -- start it!]"

    init_iss = f"Here is Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}"
    prompt_input = [
      init_iss,
      init_persona.scratch.name,
      retrieved_str,
      prev_convo_insert,
      curr_location,
      curr_context,
      init_persona.scratch.name,
      target_persona.scratch.name,
      convo_str,
      init_persona.scratch.name,
      target_persona.scratch.name,
    ]
    return prompt_input

  def __chat_func_clean_up(gpt_response: ChatUtterance, prompt=""):
    cleaned_dict = {
      "utterance": gpt_response.utterance,
      "end": gpt_response.did_conversation_end,
    }
    return cleaned_dict

  def __chat_func_validate(gpt_response, prompt=""):
    try:
      if not isinstance(gpt_response, ChatUtterance):
        return False
      return True
    except:
      traceback.print_exc()
      return False

  def get_fail_safe():
    cleaned_dict = {
      "utterance": "...",
      "end": False,
    }
    return cleaned_dict

  print("11")
  prompt_input = create_prompt_input(
    maze, init_persona, target_persona, retrieved, curr_context, curr_chat
  )
  print("22")
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  print(prompt)
  fail_safe = get_fail_safe()
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    ChatUtterance,
    repeat=3,
    fail_safe_response=fail_safe,
    func_validate=__chat_func_validate,
    func_clean_up=__chat_func_clean_up,
    verbose=verbose,
  )
  print(output)

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 4096,
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
