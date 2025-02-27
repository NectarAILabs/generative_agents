import datetime
import asyncio
import sys
sys.path.append('../../')
from utils import debug
from persona.prompt_template.run_gpt_prompt import (
    run_gpt_prompt_event_triple,
    run_gpt_prompt_event_poignancy,
    run_gpt_prompt_chat_poignancy,
    run_gpt_prompt_generate_new_schedule,
    run_gpt_prompt_planning_thought_on_convo,
    run_gpt_prompt_memo_on_convo,
)
from persona.prompt_template.gpt_structure import get_embedding
from persona.cognitive_modules.retrieve import new_retrieve

async def generate_new_schedule_on_convo(persona, statement, start_hour):
  if debug: print ("GNS FUNCTION: <generate_new_schedule_on_convo>")
  return (await run_gpt_prompt_generate_new_schedule(persona, statement, start_hour))[0]

async def sync_plan(init_persona,target_persona):
  if debug: print ("SYNC PLAN FUNCTION: <sync_plan>")
  
