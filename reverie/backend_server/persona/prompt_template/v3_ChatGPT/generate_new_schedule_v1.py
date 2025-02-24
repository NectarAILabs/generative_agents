import traceback
from typing import Any
from ..common import openai_config,get_prompt_file_path
from ..gpt_structure import ChatGPT_safe_generate_structured_response
from ..print_prompt import print_run_prompts
from pydantic import BaseModel
from datetime import timedelta
class Activity(BaseModel):
  datetime: str
  activity: str


class HourlySchedule(BaseModel):
  hourly_schedule: list[Activity]

def create_prompt(prompt_input: dict[str, Any]):
  date = prompt_input["date"]
  init_persona_currently = prompt_input["init_persona_currently"]
  hourly_schedule_org_str = prompt_input["hourly_schedule_org"]
  init_persona_name = prompt_input["init_persona_name"],
  convo_schedule_memory = prompt_input["convo_schedule_memory"],
  target_persona_name = prompt_input["target_persona_name"]
  schedule_format = prompt_input["schedule_format"]
  prompt = f"""
Current Date: {date}
Currently: {init_persona_currently}
This is the current hourly schedule for {init_persona_name} until now:
{hourly_schedule_org_str}
But after the conversation with {target_persona_name}, {init_persona_name} should change his schedule for the rest of the day according to the statements below.
Statement: {convo_schedule_memory}
===
Finishing the schedule for the rest of the day (from the next hour to 11:00 PM), follow the hourly schedule format. :
{schedule_format}
===
"""
  return prompt

# Generate new hourly schedule for the rest of the day after conversation 
# not use yet, will add to the plan soon
async def run_gpt_prompt_generate_new_schedule(
  persona, target_persona, statements, curr_context, test_input=None, verbose=False
):
  def create_prompt_input(
    persona, target_persona, convo_schedule_memory, test_input=None
  ):
    curr_date_str = persona.scratch.get_str_curr_date_str()
    curr_time = persona.scratch.curr_time
    hourly_schedule_org = persona.scratch.f_daily_schedule_hourly_org
    hourly_schedule_org_str = ""
    curr_hours_schedule = 0
    for act,dur in hourly_schedule_org:
      end_hour += curr_hours_schedule + int(dur/60)
      if end_hour <int(curr_time.hour):
        for i in range(curr_hours_schedule, end_hour):
          if i <= 12:
            hour_str = f"{i}:00 AM"
          else:
            hour_str = f"{i-12}:00 PM"
          hourly_schedule_org_str += f"[{curr_date_str} -- {hour_str}] Activity: {act}\n"
      curr_hours_schedule = end_hour

    schedule_format = '{"hourly_schedule": ['
    for i in range(1,3):
      hour = (curr_time + timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
      hour_str = hour.strftime("%I:%M %p")
      if hour.datetime.day == curr_time.datetime.day:
        schedule_format += f'{{"datetime":"{curr_date_str}, {hour_str}",'
        schedule_format += '"activity":"<to_be_determined>"},'
    prompt_input = {
      "date": persona.scratch.get_str_curr_date_str(),
      "init_persona_currently": persona.scratch.currently,
      "convo_schedule_memory": convo_schedule_memory,
      "hourly_schedule_org_str": hourly_schedule_org_str,
      "init_persona_name": persona.scratch.name,
      "target_persona_name": target_persona.scratch.name,
      "schedule_format": schedule_format,
    }
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
  def __chat_func_clean_up(gpt_response: HourlySchedule, prompt=""):
    activities = []
    for item in gpt_response.hourly_schedule:
        activity = item.activity.strip("[]")
        activity = activity.removeprefix(persona.scratch.get_str_firstname()).strip()
        activity = activity.removeprefix("is ")
        activities += [activity]
    return activities

  def __chat_func_validate(gpt_response, prompt=""):
    try:
      __chat_func_clean_up(gpt_response, prompt)
      return True
    except Exception as e:
      print("Validation failed: ", e)
      traceback.print_exc()
      return False

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
  provider_parameter = openai_config.get("other_providers", {}).get("new_hourly_schedule_provider", None)
  if provider_parameter != None:
    gpt_param.update({k:v for k,v in provider_parameter.items() if k != "model"})
    gpt_param["engine"] = provider_parameter["model"]
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(persona, target_persona, statements, curr_context)
  prompt = create_prompt(prompt_input)
  example_output = "Jane Doe is working on a project"
  fail_safe = get_fail_safe()
  output = await ChatGPT_safe_generate_structured_response(
    prompt,
    HourlySchedule,
    example_output,
    "",
    3,
    fail_safe,
    __chat_func_validate,
    __chat_func_clean_up,
    True,
  )

  if verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  if output:
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
