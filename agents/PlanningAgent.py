from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.tools import tool
#from langchain_openai import ChatOpenAI
from agents.BaseAgent import BaseAgent, Task, AgentState
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
import time
import regex
import json
import re
import torch
import outlines
from transformers import StoppingCriteria, StoppingCriteriaList

'''
prompt = ChatPromptTemplate.from_messages([
            ("system", """ Let's think step by step, and lets not repeat the prompt  :
Please determine:

1. Does the input include **concrete, factual details** (e.g., names, dates, events, locations) that are sufficient for storyboard generation?
2. Would it be **better to search the web or summarize a video/article** to enrich or verify the context?
3. Is the **user's goal and intention clearly expressed**, or is another round of clarification necessary?
4. Do I currently have sufficient knowledge in the state to generate a coherent and logical narrative for the storyboard?

Your job is to:
1. Carefully assess if the input has sufficient **substance and specificity** to support high-quality storyboard generation *without* needing to search the web, summarize external content, or clarify user intent.
2. Be pragmatic—if you could start designing a reasonable storyboard based purely on the query, select `"ready_to_design"`. If **any important gaps** exist in the user's query (e.g., ambiguity, lack of real-world grounding, missing structure or detail), select `"needs_info"`.

Decide whether the input is sufficient to design a coherent storyboard **without** external search or clarification.

Decision rule:
- If the user query and/or {knowledge} contain enough specific details (e.g., concrete entities, actions, times, constraints) to start designing now, choose "ready_to_design". Furthermore, if the input also contains sufficient visual information for the generation of an entire storyboard, then choose "ready_to_design"
- Otherwise choose "needs_info". If {knowledge} is empty or vague, default to "needs_info" unless the query itself is clearly sufficient.

Formatting rules (no deviations):
- Return output ONLY inside <final>...</final>.
- Return the object in a json forrmat as shown as well, wrapped in ```json tags
- No other text. Do NOT repeat or quote this prompt. Do NOT include markdown fences.

Example : 
<final>
```json
{{
  "status": "ready_to_design",
  "explanation": "Brief reason for your decision"
}}


</final>
             """
),("user", "Here is the user's query:\n {query}")
        ])

'''
'''
Some examples of what you can possibly return : 
```json
// Example 1
{{
  "status": "ready_to_design",
  "explanation": "The user specified 4 clear stages with a focused topic, and I possess sufficient contextual knowledge based on my training to immediately provide enough information, or the input contains enough information to generate a coherent storyboard"
}}

// Example 2
{{
  "status": "needs_info",
  "explanation": "The prompt is too vague and lacks target details, I would need to search the web for more information or the input modality was a video and I need a video analyzer to parse and understand the video"
}}
'''

#For structured output 

class PlanningDecision(BaseModel):
    rationale: str = Field(
        description="1–2 sentences explaining the decision",
        min_length=10, max_length=320,
    )
    status: Literal["ready_to_design", "needs_info"]
    needs: List[str] = Field(
        min_length = 1,
        default_factory=list,
        description=(
            "If status=needs_info, list 3–5 specific missing aspects needed "
            "to proceed "
            "Each item should be short (2–6 words), and should be "
            "something specific that the information gatherer can search for. "
        )
    )
    
    
class PlanningAgent(BaseAgent) :
    agents: Dict[str, Any] = {}
    tokenizer: Any = Field(default=None, exclude=True) 
    model: Any = Field(default=None, exclude=True)   
    

    def __init__(self, state_manager=None,**data) : 
        super().__init__(state_manager=state_manager, **data)
        
    def chat_prompt_to_string(self, chat_prompt):    
        formatted_parts = []
        for msg in chat_prompt.messages:
            formatted_parts.append(f"{msg.type.capitalize()}: {msg.content}")

        return "\n".join(formatted_parts)

    def extract_last_json_block(self, text: str, skip_first=True):
        # 1) Match fenced blocks, with or without "json" tag
        json_blocks = re.findall(r"```(?:json)?\s*\n({[\s\S]*?})\s*\n```", text, re.DOTALL)
        # 2) If none, fall back to standalone {...} but only top-level braces
        if not json_blocks:
            json_blocks = re.findall(r"\n({\s*(?:\"status\".*?\"explanation\".*?)})", text, re.DOTALL)

        # 3) Drop the example format block if asked
        if skip_first and len(json_blocks) > 1:
            json_blocks = json_blocks[1:]

        # 4) Try parsing in reverse order
        for block in reversed(json_blocks):
            try:
                parsed = json.loads(block)
                # Loose key check
                if isinstance(parsed, dict) and parsed.get("status") and parsed.get("explanation"):
                    return parsed
            except json.JSONDecodeError:
                continue

        return None



    def register_agent(self, agent_name, agent) :
        self.agents[agent_name] = agent
        self.log(f"Registered agent: {agent_name}")
        
    def devise_plan(self, initial_input, knowledge="", search_history): 
        print("Devising plan...\n")
        print(f"Search history is {search_history}")
        model = outlines.from_transformers(self.model, self.tokenizer)
        print("--------------------STARTING ORCHESTRATOR DECISION-------------------\n")
        system = (
            "You are the PlanningAgent. Decide if the input is sufficient to design a storyboard "
            "without external search or clarification. Return a 1–2 sentence rationale and a status."
            "\nRules:\n"
            "- If the query+knowledge contain enough concrete specifics (entities, actions, time/constraints, AND VERY IMPORTANTLY IMAGES/VIDEOS (file paths to them)), "
            "If there are no file paths to images or videos, always return needs_info to prompt the agent to search for Youtube videos and extract images from them"
            "choose ready_to_design; else needs_info (default if knowledge is empty/vague)."
        )
        prompt = (
            f"{system}\n\n"
            f"Query:\n{initial_input}\n\n"
            f"Knowledge:\n{knowledge or '(empty)'}\n"
            f"Search History:\n {search_history}\n"
            """
            Return only rationale and status. Relate the rationale to the initial_input and explain what needs to be searched more for a more coherent storyboard. Include whether the information can sufficiently capture the nuance of the input query and requirements. Then if the current information currently lacks some form of nuance capturing, tell the information_gatherer what it should search for to achieve the goal.
            """
            """
            If status = needs_info, identify at least 3 specific missing aspects the agent must gather (e.g., desired tone, target audience, key visuals, or specific parade highlights etc.) Return them in the needs field as a JSON list of short phrases. If status = "needs_info", return needs as 3–6 specific, search-ready keywords/short phrases (2–6 words each) that directly represent whatever the query is asking for. 
Forbidden (reject): “desired tone”, “target audience”, “parade highlights”, “more context”, “visuals”.
Just as a general list : POV walkthrough
entrance sequence
finale fireworks
crowd wave shot
mascot appearance
volunteer stories
family-friendly shots
accessibility features
eco initiatives
sustainability efforts
community outreach
local businesses
vendor stalls
food market scenes
night lights bokeh
golden hour shots
sunrise establishing
skyline time-lapse
rain backup plan
contingency planning

Never leave needs empty, and format it to sound similar to the list as above.

Also, refer to the search_history above and TRY NOT TO Repeat whatever has been searched before. Try expanding search radius.
Output only the JSON object.
            """
        )
        
        json_text = model(prompt, output_type=PlanningDecision, temperature=0.2, max_new_tokens=200)

        # It returns a JSON string; validate/parse with Pydantic:
        decision = PlanningDecision.model_validate_json(json_text)
        # print(f"Decision is {decision} and is of type {type(decision)}\n")

        print("Rationale:", decision.rationale)
        print("Status:", decision.status)
        print("Needs:" , decision.needs)
        print("===========================END OF RESPONSE FOR ORCHESTRATOR========================\n")
        return decision
#         formatted_prompt = prompt.invoke({"query": initial_input,
#                                          "knowledge": knowledge})
#         formatted_prompt = self.chat_prompt_to_string(formatted_prompt)
#         # print("\n\n\n Formatted Prompt is here : ")
#         # print(formatted_prompt)
#         print("\n\n\n")
#         inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
#         inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
#         if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token
            
#         print("Thinking......")
#         # print(f"Model is on device: {self.model.device}")
#         # for name, tensor in inputs.items():
#         #     print(f"Input '{name}' is on device: {tensor.device}")
#         with torch.no_grad():
#             outputs = self.model.generate(**inputs, temperature=0.2, max_new_tokens=128, stop=["</final>"])
#         print("Decoding...")
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         print("Response : ")
#         generated_response = response[len(formatted_prompt):]
#         print(generated_response)
#         print("----------------END OF RESPONSE------------------\n")

#         new_state = self.state_manager.model_copy(deep=True)

#         try:
#             plan_data = self.extract_last_json_block(generated_response, skip_first=True)

#             if not plan_data:
#                 raise ValueError("Could not extract valid JSON from response.")

#             new_state.add_to_conversation("system", f"Created plan: {plan_data['explanation']}")
#             new_state.next_step = "execute"
            
#             print(f"\n\nthe new state i have in the planning agent devise plan is {new_state}\n\n")
#             print(f"the plan data i have in the planning agent devise plan is {plan_data}\n\n")

#             return new_state, plan_data

#         except Exception as e:
#             print(f"Error parsing plan!!: {e}")
#             print("Raw model output:")
#             print(response)

#             new_state = self.state_manager.model_copy(deep=True)
#             new_state.add_to_conversation("system", "Could not parse plan or response format was invalid.")
#             new_state.next_step = "execute"

#             return new_state, {
#                 "status": "needs_info",
#                 "explanation": "Could not parse plan or response format was invalid."
#             }

    def get_plan(self) : 
        return self.get_state().get('workflow_plan', [])

    def confirm_plan(self) :
        plan = self.get_plan()

        print("Here is the proposed plan for now : ")
        print(plan)

        confirmation = input("Do you confirm this plan? (yes/no): ").strip().lower()

        if confirmation == "yes":
            self.log("Plan confirmed by the user.")
            self.update_state(plan_confirmation=True)
            new_state.next_step = 'execute'
            return True
        else:
            self.log("Plan not confirmed by the user.")
            return False

    def execute_plan(self):
        plan = self.get_plan()

if __name__ == "__main__":
    user_query1="Can you generate a storyboard about the anime Ruruoni Kenshin?"

    agent_state = AgentState(user_query=user_query1)
    planning_agent = PlanningAgent(state_manager=agent_state)
    print("loaded agent")
    assert planning_agent.get_state().next_step == "plan", "Initial next_step should be 'plan'"
    assert isinstance(planning_agent, BaseAgent), "PlanningAgent should inherit from BaseAgent"

    planning_agent.log("Testing testing!!")
    history = planning_agent.get_state().conversation_history
    print(history)

    devised_plan = planning_agent.devise_plan(user_query1)
    print("Here is the Devised plan : ------------------------------\n")
    print(devised_plan)
