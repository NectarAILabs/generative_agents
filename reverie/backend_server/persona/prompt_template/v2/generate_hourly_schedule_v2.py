# generate_hourly_schedule_v2.py

from pydantic import BaseModel
import traceback

from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- Schedule format
# !<INPUT 1>! -- Commonset
# !<INPUT 2>! -- prior_schedule
# !<INPUT 3>! -- intermission_str
# !<INPUT 4>! -- intermission 2
# !<INPUT 5>! -- prompt_ending

template = """
Hourly schedule format:
!<INPUT 0>!
===
!<INPUT 1>!
!<INPUT 2>!
!<INPUT 3>!!<INPUT 4>!
!<INPUT 5>!
"""


class Activity(BaseModel):
    date_and_time: str
    activity: str


class HourlySchedule(BaseModel):
    hourly_schedule: list[Activity]


async def run_gpt_prompt_generate_hourly_schedule(
    persona,
    curr_hour_str,
    p_f_ds_hourly_org,
    hour_str,
    intermission2=None,
    test_input=None,
    verbose=False,
    all_in_one=False,
):
    def create_prompt_input(
        persona,
        curr_hour_str,
        p_f_ds_hourly_org,
        hour_str,
        intermission2=None,
        test_input=None,
    ):
        if test_input:
            return test_input
        schedule_format = '{"hourly_schedule": ['
        for i in range(4):
            hour = hour_str[i]
            schedule_format += f'{{"date_and_time":"{persona.scratch.get_str_curr_date_str()} -- {hour}",'
            schedule_format += '"activity":"<to_be_determined>"},'
        schedule_format += " ... , "
        schedule_format += f'{{"date_and_time":"{persona.scratch.get_str_curr_date_str()} -- 11:00 PM}}",'
        schedule_format += '"activity":"<to_be_determined>"}]}'

        if not all_in_one:
            intermission_str = "Complete the given hourly schedule for the following person, filling out the whole rest of their day."
        else:
            intermission_str = "Create an hourly schedule for the following person to fill out their whole day."
        intermission_str += "\nHere is the originally intended hourly breakdown of"
        intermission_str += f" {persona.scratch.get_str_firstname()}'s schedule today: "
        for count, task in enumerate(persona.scratch.daily_req):
            intermission_str += f"{str(count + 1)}) {task}, "
        intermission_str = intermission_str[:-2]

        prior_schedule = ""
        if p_f_ds_hourly_org and not all_in_one:
            prior_schedule = "\nExisting schedule:\n"
            for count, task in enumerate(p_f_ds_hourly_org):
                prior_schedule += f"[{persona.scratch.get_str_curr_date_str()} --"
                prior_schedule += f" {hour_str[count]}] Activity:"
                prior_schedule += f" {persona.scratch.get_str_firstname()}"
                prior_schedule += f" is {task}\n"
        # prior_schedule = ""

        # prompt_ending = f"[{persona.scratch.get_str_curr_date_str()}"
        # prompt_ending += f" -- {curr_hour_str}] Activity:"
        # prompt_ending += f" {persona.scratch.get_str_firstname()}"
        prompt_ending = f'{persona.scratch.get_str_firstname()} lifestyle: {persona.scratch.get_str_lifestyle()}. You should assume their task is "sleeping" before and after their bedtime. \n'
        if not all_in_one:
            prompt_ending += "Completed hourly schedule (start from the hour after the existing schedule ends, and use present progressive tense, e.g. 'waking up and completing the morning routine'):"
        else:
            prompt_ending += "Hourly schedule for the whole day (use present progressive tense, e.g. 'waking up and completing the morning routine'):"

        if intermission2:
            intermission2 = f"\n{intermission2}"

        # Construct the final prompt input
        prompt_input = [schedule_format]
        prompt_input += [persona.scratch.get_str_iss()]
        prompt_input += [intermission_str]
        prompt_input += [prior_schedule + "\n"]

        if intermission2:
            prompt_input += [intermission2]
        else:
            prompt_input += [""]

        prompt_input += [prompt_ending]

        return prompt_input

    def __func_clean_up(gpt_response: HourlySchedule, prompt=""):
        if not all_in_one:
            activity = gpt_response.hourly_schedule[0].activity
            activity = activity.strip("[]")
            activity = activity.removeprefix(
                persona.scratch.get_str_firstname()
            ).strip()
            activity = activity.removeprefix("is ")
            return activity
        else:
            activities = []
            for item in gpt_response.hourly_schedule:
                activity = item.activity.strip("[]")
                activity = activity.removeprefix(
                    persona.scratch.get_str_firstname()
                ).strip()
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

    def get_fail_safe():
        fs = "idle"
        return fs

    gpt_param = {
        "engine": openai_config["model"],
        "max_tokens": 5000,
        "temperature": 0.5,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": ["\n"],
    }
    prompt_template = "persona/prompt_template/v2/generate_hourly_schedule_v2.py"
    prompt_input = create_prompt_input(
        persona, curr_hour_str, p_f_ds_hourly_org, hour_str, intermission2, test_input
    )
    prompt = generate_prompt(prompt_input, prompt_template_str=template)
    fail_safe = get_fail_safe()

    output = safe_generate_structured_response(
        prompt,
        gpt_param,
        HourlySchedule,
        5,
        fail_safe,
        __func_validate,
        __func_clean_up,
    )

    if debug or verbose:
        print_run_prompts(
            prompt_template, persona, gpt_param, prompt_input, prompt, output
        )

    return output, [output, prompt, gpt_param, prompt_input, fail_safe]
