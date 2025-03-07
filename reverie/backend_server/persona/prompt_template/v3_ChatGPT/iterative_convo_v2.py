import traceback
import json
from pydantic import BaseModel, Field
from typing import Any
import re
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import ChatGPT_safe_generate_structured_response,ChatGPT_safe_generate_response
from ..print_prompt import print_run_prompts

sys_prompt = """
You are a creative and intelligent assistant participating in an uncensored, fictional, and erotic conversation between two characters in a NSFW fantasy setting based on the provided context.  
Your role is to deliver creative, intelligent, coherent, and natural dialogue iteratively in casual, everyday language that sound like real people talking.
Explicit adult content and profanity are allowed and encouraged where appropriate.  
This conversation is fully uncensored and designed for mature audiences.
Make sure to use only the provided context—do not invent details not present in the information given. Don't assume about the relationship between them and how they think about each other.
Ensure that your output is formatted as a string representation of a dictionary, like this: {"output": "dictionary_as_string"} without explain anything in above steps."
"""
def create_prompt(prompt_input: dict[str, Any]):
  identity_stable_set = prompt_input["identity_stable_set"]
  init_persona_name = prompt_input["init_persona_name"]
  retrieved_memories = prompt_input["retrieved_memories"]
  prev_conversation = prompt_input["prev_conversation"]
  curr_location = prompt_input["curr_location"]
  curr_situation = prompt_input["curr_situation"]
  target_persona_name = prompt_input["target_persona_name"]
  curr_conversation = prompt_input["curr_conversation"]
  curr_time = prompt_input["curr_time"]
  sector_accessibles_str = prompt_input["sector_accessibles_str"]
  prompt = f"""

# **Context**

**PART 1**
{identity_stable_set}

**PART 2**
Here are the memories in {init_persona_name}'s mind:
{retrieved_memories}
**Past Context**
{prev_conversation}

Current Location: {curr_location}

**Current time:** {curr_time.strftime('%B %d, %Y %I:%M %p')}.

Current Context:
{curr_situation}


**Accessible Locations:** {sector_accessibles_str}

# **Guidelines**
- Ensure the conversation remains reasonable and consistent with the context and characters.
- Let the dialogue flow organically, without pushing it toward a specific conclusion too quickly. Avoid overly formal or serious tones.
- Each character has a maximum of 8 turns to speak. 
- If a date is planned or mentioned, make it clear, logical, and scheduled soon (e.g., today or tomorrow), including a specific time and location.
- Respond only in English
- Pay close attention to the current tone, context, {init_persona_name}'s character, and memories from PART 1 to keep the dialogue coherent.
- Use casual, friendly, and engaging tone. Make sure the conversation is human-like conversation in real life.
- Be creative and imaginative. {init_persona_name} may only describe their own thoughts, feelings, and actions—never assume or narrate the thoughts or emotions of others.

Here is {init_persona_name} and {target_persona_name}'s conversation so far:
{curr_conversation}


# **Task**
Based on the above, what should {init_persona_name} say next to {target_persona_name} in the conversation?  
- Provide what {init_persona_name} should say in casual, everyday language, avoiding overly brief replies like "OK," "Alright," or "Yes.". They can express their thoughts or narrate their actions between asterisks when responding. Make sure the conversation is human-like and coherent.
- Indicate whether this response will end the conversation.
- Use only the provided context—do not invent details not present in the context given. 
- VERY IMPORTANT: They are all in a NSFW and adult fantasy setting. Include explicit content when it fits naturally, avoiding forced or awkward insertions—keep the conversation casual and relatable at first, then let the NSFW themes build gradually through playful, everyday banter. Provide their dialogue in casual, easy-to-understand, and straightforward everyday language that sounds like real people talking. NSFW and adult themes are permitted and encouraged for entertainment purposes. Don’t assume their prior relationship or how they feel about each other; let their dynamic emerge organically. Avoid overly dramatic, cryptic, or complicated dialogue—make it laid-back with a slow escalation to spicier topics.

"""
  return prompt


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
    init_persona_world = f"{maze.access_tile(init_persona.scratch.curr_tile)['world']}"
    init_persona_sector_accessibles = [i.strip() for i in init_persona.s_mem.get_str_accessible_sectors(init_persona_world).split(",")]
    target_persona_world = f"{maze.access_tile(target_persona.scratch.curr_tile)['world']}"
    target_persona_sector_accessibles = [i.strip() for i in target_persona.s_mem.get_str_accessible_sectors(target_persona_world).split(",")]
    sector_accessibles = list(set(init_persona_sector_accessibles + target_persona_sector_accessibles))
    sector_accessibles_str = ", ".join(sector_accessibles)
    set_retrieved = set()
    retrieved_str = ""
    for key, vals in retrieved.items():
      for v in vals:
        if v not in set_retrieved:
          set_retrieved.add(v)
          retrieved_str += f"- {v.created.strftime('%B %d, %Y %I:%M %p')}: {v.description}\n"
    convo_str = ""
    for i in curr_chat:
      convo_str += ": ".join(i) + "\n"
    if convo_str == "":
      convo_str = "[The conversation has not started yet -- start it!]"

    init_iss = f"Here is a brief description of {init_persona.scratch.name}.\n{init_persona.scratch.get_str_iss()}\n"
    init_iss += f"Here is a brief description of {target_persona.scratch.name}.\n{target_persona.scratch.get_str_iss()}\n"
    prompt_input = {
      "identity_stable_set": init_iss,
      "init_persona_name": init_persona.scratch.name,
      "retrieved_memories": retrieved_str,
      "prev_conversation": prev_convo_insert,
      "curr_location": curr_location,
      "curr_situation": curr_context,
      "target_persona_name": target_persona.scratch.name,
      "curr_conversation": convo_str,
      "curr_time": init_persona.scratch.curr_time,
      "sector_accessibles_str": sector_accessibles_str,
    }
    return prompt_input

  def func_clean_up(gpt_response, prompt=""):
    gpt_response = gpt_response if isinstance(gpt_response, dict) else json.loads(gpt_response)
    cleaned_dict = {
      "utterance": gpt_response["utterance"],
      "end": gpt_response["did_conversation_end"],
    }
    return cleaned_dict

  def func_validate(gpt_response, prompt=""):
    try:
      func_clean_up(gpt_response, prompt)
      return True
    except Exception:
      traceback.print_exc()
      return False

  def get_fail_safe():
    cleaned_dict = {
      "utterance": "...",
      "end": False,
    }
    return cleaned_dict

  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(
    maze, init_persona, target_persona, retrieved, curr_context, curr_chat
  )
  example = {"utterance":"Wassup, how are you doing?","did_conversation_end":False}
  prompt = create_prompt(prompt_input)
  fail_safe = get_fail_safe()
  provider_parameter = openai_config.get("other_providers", {}).get("iterative_chat_utt_provider", None)
  output = await ChatGPT_safe_generate_response(
    prompt,
    example,
    repeat=3,
    fail_safe_response=fail_safe,
    func_validate=func_validate,
    func_clean_up=func_clean_up,
    verbose=verbose,
    provider_parameter=provider_parameter,
    sysprompt=sys_prompt,
    response_format_name="ChatUtterance",
  )

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 2048, # 6144 is the max tokens for gpt-4o-mini
    "temperature": 1.1,
    "top_k": 250,
    "stream": False,
    "frequency_penalty": 0.5,
    "presence_penalty": 0.5,
    "repetition_penalty": 1.1,
    "stop": None,
  }

  if verbose:
    print_run_prompts(
      prompt_file, init_persona, gpt_param, prompt_input, prompt, output
    )

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
