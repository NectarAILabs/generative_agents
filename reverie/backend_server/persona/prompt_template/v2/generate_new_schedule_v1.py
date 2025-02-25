import traceback
from typing import Any
from ..common import openai_config,get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts
from pydantic import BaseModel
from datetime import timedelta
class Activity(BaseModel):
  datetime: str
  activity: str


class HourlySchedule(BaseModel):
  hourly_schedule: list[Activity]

def create_prompt(prompt_input: dict[str, Any]):
  init_persona_currently = prompt_input["init_persona_currently"]
  hourly_schedule_org_str = prompt_input["hourly_schedule_org_str"]
  init_persona_name = prompt_input["init_persona_name"]
  init_persona_lifestyle = prompt_input["init_persona_lifestyle"]
  convo_schedule_memory = prompt_input["convo_schedule_memory"]
  schedule_format = prompt_input["schedule_format"]
  start_hour_str = prompt_input["start_hour_str"]
  prompt = f"""
Currently: {init_persona_currently}
Lifestyle: {init_persona_lifestyle}
This is the current hourly schedule for {init_persona_name} until now:
{hourly_schedule_org_str}
But after the conversation, {init_persona_name} should change his/her schedule for the rest of the day according to the statement below.
Statement: {convo_schedule_memory}
===
Finishing the schedule for the rest of the day (start from {start_hour_str} to the end of the day), follow the format:
{schedule_format}
Your schedule should assume that their task is ONLY "sleeping" after their bedtime and before they wake up.
Only modify the schedule when a statement explicitly refers to a specific appointment. Do not create or assume a schedule based on uncertain or unspecified plans with particular individuals. The action should be clear and easy to understand.
===
"""
  return prompt

# Generate new hourly schedule for the rest of the day after conversation 
# receive the planning_thought as statement from conversation, replan the other activities in the schedule of the rest of the day
async def run_gpt_prompt_generate_new_schedule(
  persona, statement, start_hour,test_input=None, verbose=False
):
  def create_prompt_input(
    persona, statement, start_hour, test_input=None
  ):
    curr_date_str = persona.scratch.get_str_curr_date_str()
    hourly_schedule_org = persona.scratch.f_daily_schedule_hourly_org
    hourly_schedule_org_str = ""
    _hours_schedule = 0
    for act,dur in hourly_schedule_org:
      for i in range(_hours_schedule, _hours_schedule + int(dur/60)):
        if i < 12:
          hour_str = f"{i}:00 AM"
        elif i==12:
          hour_str = f"{i}:00 PM"
        else:
          hour_str = f"{i-12}:00 PM"
        hourly_schedule_org_str += f"[{curr_date_str} -- {hour_str}] Activity: {act}\n"
      _hours_schedule += int(dur/60)

    schedule_format = '{"hourly_schedule": ['
    start_hour_str = persona.scratch.curr_time.replace(hour=start_hour,minute=0,second=0,microsecond=0).strftime("%I:%M %p")
    for i in range(0,3):
      hour = (persona.scratch.curr_time.replace(hour=start_hour,minute=0,second=0,microsecond=0) + timedelta(hours=i))
      if hour.day == persona.scratch.curr_time.day:
        hour_str = hour.strftime("%I:%M %p")
        schedule_format += f'{{"datetime":"{curr_date_str}, {hour_str}",'
        schedule_format += '"activity":"<to_be_determined>"},'
    schedule_format += f'{{"...datetime":"{curr_date_str}, 11:00 PM",'
    prompt_input = {
      'init_persona_name': persona.scratch.name,
      "init_persona_currently": persona.scratch.get_str_currently(),
      "convo_schedule_memory": statement,
      "hourly_schedule_org_str": hourly_schedule_org_str,
      "init_persona_lifestyle": persona.scratch.get_str_lifestyle(),
      "schedule_format": schedule_format,
      "start_hour_str": start_hour_str
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
    return []

  # ChatGPT Plugin ===========================================================
  def __func_clean_up(gpt_response: HourlySchedule, prompt=""):
    activities = []
    for item in gpt_response.hourly_schedule:
        activity = item.activity.strip("[]")
        activity = activity.removeprefix(persona.scratch.get_str_firstname()).strip()
        activity = activity.removeprefix("is ")
        activities += [activity]
    return activities

  def __func_validate(gpt_response, prompt=""):
    try:
      __func_clean_up(gpt_response, prompt)
      return True
    except Exception as e:
      print("Validation failed: ", e)
      traceback.print_exc()
      return False

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 4096,
    "temperature": 0.7,
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
  prompt_input = create_prompt_input(persona, statement, start_hour)
  prompt = create_prompt(prompt_input)
  fail_safe = get_fail_safe()
  output = await safe_generate_structured_response(
    prompt,
    gpt_param,
    HourlySchedule,
    5,
    fail_safe,
    __func_validate,
    __func_clean_up,
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
