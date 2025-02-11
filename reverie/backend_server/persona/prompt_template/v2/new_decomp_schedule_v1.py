# new_decomp_schedule_v1.py

import datetime
from pydantic import BaseModel
import traceback

from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- persona name
# !<INPUT 1>! -- start hour
# !<INPUT 2>! -- end hour
# !<INPUT 3>! -- original plan
# !<INPUT 4>! -- persona name
# !<INPUT 5>! -- new event
# !<INPUT 6>! -- new event duration
# !<INPUT 7>! -- persona name
# !<INPUT 8>! -- start hour
# !<INPUT 9>! -- end hour
# !<INPUT 10>! -- end hour
# !<INPUT 11>! -- new schedule init

template = """
Here was !<INPUT 0>!'s originally planned schedule from !<INPUT 1>! to !<INPUT 2>!.
!<INPUT 3>!

But !<INPUT 4>! unexpectedly ended up !<INPUT 5>! for !<INPUT 6>! minutes. Revise !<INPUT 7>!'s schedule from !<INPUT 8>! to !<INPUT 9>! accordingly (it has to end by !<INPUT 10>!).
The revised schedule:
!<INPUT 11>!

The above activities are formatted like:
[start_time] ~ [end_time] -- [main_task] ([subtask])
"""


class NewActivity(BaseModel):
  start_time: str
  end_time: str
  main_task: str
  subtask: str


class NewSchedule(BaseModel):
  schedule: list[NewActivity]


async def run_gpt_prompt_new_decomp_schedule(
  persona,
  main_act_dur,
  truncated_act_dur,
  start_time_hour,
  end_time_hour,
  inserted_act,
  inserted_act_dur,
  test_input=None,
  verbose=False,
):
  def create_prompt_input(
    persona,
    main_act_dur,
    truncated_act_dur,
    start_time_hour,
    end_time_hour,
    inserted_act,
    inserted_act_dur,
    test_input=None,
  ):
    persona_name = persona.name
    start_hour_str = start_time_hour.strftime("%H:%M %p")
    end_hour_str = end_time_hour.strftime("%H:%M %p")

    original_plan = ""
    for_time = start_time_hour
    for i in main_act_dur:
      original_plan += (
        f"{for_time.strftime('%H:%M')} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime('%H:%M')} -- "
        + i[0]
      )
      original_plan += "\n"
      for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init = ""
    for_time = start_time_hour
    for count, i in enumerate(truncated_act_dur):
      new_plan_init += (
        f"{for_time.strftime('%H:%M')} ~ {(for_time + datetime.timedelta(minutes=int(i[1]))).strftime('%H:%M')} -- "
        + i[0]
      )
      new_plan_init += "\n"
      if count < len(truncated_act_dur) - 1:
        for_time += datetime.timedelta(minutes=int(i[1]))

    new_plan_init += (for_time + datetime.timedelta(minutes=int(i[1]))).strftime(
      "%H:%M"
    ) + " ~"

    prompt_input = [
      persona_name,
      start_hour_str,
      end_hour_str,
      original_plan,
      persona_name,
      inserted_act,
      inserted_act_dur,
      persona_name,
      start_hour_str,
      end_hour_str,
      end_hour_str,
      new_plan_init,
    ]
    return prompt_input

  def __func_clean_up(gpt_response: NewSchedule, prompt=""):
    new_schedule = []

    for activity in gpt_response.schedule:
      start_time = activity.start_time
      end_time = activity.end_time
      delta = datetime.datetime.strptime(
        end_time, "%H:%M"
      ) - datetime.datetime.strptime(start_time, "%H:%M")
      delta_min = int(delta.total_seconds() / 60)
      if delta_min < 0:
        delta_min = 0
      action = activity.main_task + f" ({activity.subtask})"
      new_schedule += [[action, delta_min]]

    return new_schedule

  def __func_validate(gpt_response, prompt=""):
    try:
      gpt_response = __func_clean_up(gpt_response, prompt)
      dur_sum = 0
      for act, dur in gpt_response:
        dur_sum += dur
        if str(type(act)) != "<class 'str'>":
          return False
        if str(type(dur)) != "<class 'int'>":
          return False
      x = (
        prompt.split("\n")[0].split("originally planned schedule from")[-1].strip()[:-1]
      )
      x = [datetime.datetime.strptime(i.strip(), "%H:%M %p") for i in x.split(" to ")]
      delta_min = int((x[1] - x[0]).total_seconds() / 60)

      if int(dur_sum) != int(delta_min):
        return False

    except:
      traceback.print_exc()
      return False
    return True

  def get_fail_safe(main_act_dur, truncated_act_dur):
    dur_sum = 0
    for act, dur in main_act_dur:
      dur_sum += dur

    ret = truncated_act_dur[:]
    ret += main_act_dur[len(ret) - 1 :]

    # If there are excess, we need to trim...
    ret_dur_sum = 0
    count = 0
    over = None
    for act, dur in ret:
      ret_dur_sum += dur
      if ret_dur_sum == dur_sum:
        break
      if ret_dur_sum > dur_sum:
        over = ret_dur_sum - dur_sum
        break
      count += 1

    if over:
      ret = ret[: count + 1]
      ret[-1][1] -= over

    return ret

  gpt_param = {
    "engine": openai_config["model"],
    "max_tokens": 10000,
    "temperature": 0,
    "top_p": 1,
    "stream": False,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "stop": None,
  }
  prompt_template = "persona/prompt_template/v2/new_decomp_schedule_v1.py"
  prompt_input = create_prompt_input(
    persona,
    main_act_dur,
    truncated_act_dur,
    start_time_hour,
    end_time_hour,
    inserted_act,
    inserted_act_dur,
    test_input,
  )
  prompt = generate_prompt(prompt_input, prompt_template_str=template)
  fail_safe = get_fail_safe(main_act_dur, truncated_act_dur)
  output = await safe_generate_structured_response(
    prompt, gpt_param, NewSchedule, 5, fail_safe, __func_validate, __func_clean_up
  )

  # print ("* * * * output")
  # print (output)
  # print ('* * * * fail_safe')
  # print (fail_safe)

  if debug or verbose:
    print_run_prompts(prompt_template, persona, gpt_param, prompt_input, prompt, output)

  return output, [output, prompt, gpt_param, prompt_input, fail_safe]
