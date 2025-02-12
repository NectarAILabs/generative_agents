# insight_and_evidence_v1.py

from pydantic import BaseModel
import traceback

from utils import debug
from ..common import openai_config
from ..gpt_structure import generate_prompt, safe_generate_structured_response
from ..print_prompt import print_run_prompts

# Variables:
# !<INPUT 0>! -- Numbered list of event/thought statements
# !<INPUT 1>! -- target persona name or "the conversation"

template = """
Input:
!<INPUT 0>!

What !<INPUT 1>! high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))
"""


class Insight(BaseModel):
    insight: str
    because_of: list[int]


class InsightGuidance(BaseModel):
    insights: list[Insight]


def run_gpt_prompt_insight_and_guidance(
    persona, statements, n, test_input=None, verbose=False
):
    def create_prompt_input(persona, statements, n, test_input=None):
        prompt_input = [statements, str(n)]
        return prompt_input

    def __func_clean_up(gpt_response: InsightGuidance, prompt=""):
        for item in gpt_response.insights:
            item.insight = item.insight.split("(because of ")[0]
        ret = {item.insight: item.because_of for item in gpt_response.insights}
        return ret

    def __func_validate(gpt_response, prompt=""):
        try:
            if not isinstance(gpt_response, InsightGuidance):
                return False
            __func_clean_up(gpt_response, prompt)
            return True
        except:
            traceback.print_exc()
            return False

    def get_fail_safe(n):
        return {"I am okay": [1, 2, 3]}

    gpt_param = {
        "engine": openai_config["model"],
        "max_tokens": 500,
        "temperature": 0.5,
        "top_p": 1,
        "stream": False,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    }
    prompt_template = "persona/prompt_template/v2/insight_and_evidence_v1.py"
    prompt_input = create_prompt_input(persona, statements, n)
    prompt = generate_prompt(prompt_input, prompt_template_str=template)

    fail_safe = get_fail_safe(n)
    output = safe_generate_structured_response(
        prompt,
        gpt_param,
        InsightGuidance,
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
