from pydantic import BaseModel
import traceback
from typing import Any

from utils import debug
from ..common import openai_config, get_prompt_file_path
from ..gpt_structure import safe_generate_structured_response
from ..print_prompt import print_run_prompts


def create_prompt(prompt_input: dict[str, Any]):
  persona_name = prompt_input["persona_name"]
  schedule_format = prompt_input["schedule_format"]
  identity_stable_set = prompt_input["identity_stable_set"]
  instructions = prompt_input["instructions"]
  broad_daily_plan = prompt_input["broad_daily_plan"]
  existing_schedule = prompt_input["existing_schedule"]
  extra_instructions = prompt_input["extra_instructions"]
  prompt_ending = prompt_input["prompt_ending"]

  prompt = f"""
{identity_stable_set}
{instructions}
Replace "[Fill in]" with the actual activity for each hour. Keep the actual activity general and not too specific. 
Focus on general activities rather than specific events or interactions. The schedule should include flexible tasks like work, exercise, meals, relaxation, and other productive or leisure activities, but avoid mentioning other people, detailed events, or things {persona_name} can't control.
Here is the originally intended hourly breakdown of {persona_name}'s schedule today:
{broad_daily_plan}
{existing_schedule}
{extra_instructions}
{prompt_ending}
===
Follow the hourly schedule format (from 00:00 AM to 11:00 PM). Remember to include all 24 hours of the day.:
{schedule_format}
===
"""
  return prompt


class Activity(BaseModel):
  datetime: str
  activity: str


class HourlySchedule(BaseModel):
  hourly_schedule: list[Activity]

async def run_gpt_prompt_generate_hourly_schedule(
  persona,
  p_f_ds_hourly_org,
  hour_strings,
  extra_instructions="",
  test_input=None,
  verbose=False,
  all_in_one=True,
):
  def create_prompt_input(
    persona,
    p_f_ds_hourly_org,
    hour_strings,
    extra_instructions="",
    test_input=None,
  ):
    if test_input:
      return test_input

    curr_date_str = persona.scratch.get_str_curr_date_str()
    persona_firstname = persona.scratch.get_str_firstname()

    schedule_format = '{"hourly_schedule": ['
    for i in range(4):
      hour = hour_strings[i]
      schedule_format += f'{{"datetime":"{curr_date_str}, {hour}",'
      schedule_format += '"activity":"<to_be_determined>"},'
    schedule_format += " ... continue ... , "
    schedule_format += f'{{"datetime":"{curr_date_str}, 11:00 PM"}}",'
    schedule_format += '"activity":"<to_be_determined>"}]}'
    
    if all_in_one:
      instructions = "Create an hourly schedule for the following person to fill out their whole day."
    else:
      instructions = "Complete the given hourly schedule for the following person, filling out the whole rest of their day."

    broad_daily_plan = ""
    for count, task in enumerate(persona.scratch.daily_req):
      broad_daily_plan += f"{str(count + 1)}) {task}, "
    broad_daily_plan = broad_daily_plan[:-2]

    existing_schedule = ""
    if p_f_ds_hourly_org and not all_in_one:
      existing_schedule = "\nExisting schedule:\n"
      for count, task in enumerate(p_f_ds_hourly_org):
        existing_schedule += f"[{curr_date_str} --"
        existing_schedule += f" {hour_strings[count]}] Activity:"
        existing_schedule += f" {persona_firstname}"
        existing_schedule += f" is {task}\n"
    prompt_ending = f'{persona.scratch.get_str_firstname()} lifestyle: {persona.scratch.get_str_lifestyle()}\nYour schedule should assume that their task is ONLY "sleeping" after their bedtime and before they wake up. For example if they go to sleep at 2am and wake up at 11am, your schedule should make sure that the action from 02:00 AM to 11:00 AM is sleeping\n'
    if all_in_one:
      prompt_ending += "Hourly schedule for the whole day (use present progressive tense, e.g. 'waking up and completing the morning routine'):"
    else:
      prompt_ending += "Completed hourly schedule (start from the hour after the existing schedule ends, and use present progressive tense, e.g. 'waking up and completing the morning routine'):"

    prompt_input = {
      "persona_name": persona_firstname,
      "schedule_format": schedule_format,
      "identity_stable_set": persona.scratch.get_str_iss(),
      "instructions": instructions,
      "broad_daily_plan": broad_daily_plan,
      "existing_schedule": existing_schedule,
      "extra_instructions": extra_instructions,
      "prompt_ending": prompt_ending,
    }

    return prompt_input

  def __func_clean_up(gpt_response: HourlySchedule, prompt=""):
    if all_in_one:
      activities = []
      for item in gpt_response.hourly_schedule:
        activity = item.activity.strip("[]")
        activity = activity.removeprefix(persona.scratch.get_str_firstname()).strip()
        activity = activity.removeprefix("is ")
        activities += [activity]
      return activities
    else:
      activity = gpt_response.hourly_schedule[0].activity
      activity = activity.strip("[]")
      activity = activity.removeprefix(persona.scratch.get_str_firstname()).strip()
      activity = activity.removeprefix("is ")
      return activity

  def __func_validate(gpt_response, prompt=""):
    try:
      __func_clean_up(gpt_response, prompt)
      return True
    except Exception as e:
      print("Validation failed: ", e)
      traceback.print_exc()
      return False

  def get_fail_safe():
    if all_in_one:
      fs = ["idle"] * len(hour_strings)
    else:
      fs = "idle"
    return fs
  gpt_param = {
      "engine": openai_config["model"],
      "max_tokens": 5000,
      "temperature": 0,
      "top_p": 1,
      "stream": False,
      "frequency_penalty": 0,
      "presence_penalty": 0,
      "stop": ["\n"],
    }
  provider_parameter = openai_config.get("other_providers", {}).get("hourly_schedule_provider", None)
  if provider_parameter != None:
    gpt_param.update({k:v for k,v in provider_parameter.items() if k != "model"})
    gpt_param["engine"] = provider_parameter["model"]
  prompt_file = get_prompt_file_path(__file__)
  prompt_input = create_prompt_input(
    persona, p_f_ds_hourly_org, hour_strings, extra_instructions, test_input
  )
  prompt = create_prompt(prompt_input)
  fail_safe = get_fail_safe()

  output = await safe_generate_structured_response(
    prompt, gpt_param, HourlySchedule, 5, fail_safe, __func_validate, __func_clean_up
  )

  if debug or verbose:
    print_run_prompts(prompt_file, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
