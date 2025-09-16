from agents.PlanningAgent import PlanningAgent
from agents.BaseAgent import AgentState

def orchestrator(model, tokenizer, state: dict) -> dict:
    
    print("Routed back to the Orchestrator Note")
    print(state)
    input = state["input"]
    knowledge = state.get("knowledge")

    planning_agent = PlanningAgent(model=model, 
                                   tokenizer=tokenizer,
                                   state_manager=state)
    # Debugging print statements
    # if knowledge :
    #     print("Knowledge exists\n")
    #     print(f"Hence the input is {input} and the knowledge we have is {knowledge}")
    # else : 
    #     print("Knowledge doesn't exist\n")
    #     print(f"Hence the input is {input}")
        
    decision = planning_agent.devise_plan(input,knowledge, state['search_history'])
    # Debug Print statements
    # print(f"\nResult status is : {decision.status}\n")
    # print(f"Rationale behind that is : {decision.rationale}\n")
    # print(f"Needs  is : {decision.needs}\n")
    
    status = decision.status
    rationale = decision.rationale
    needs = decision.needs
    # print(f"Current Search History is {state['search_history']}\n")
    

    return {
        "reason_for_retry": rationale,
        "status": status,
        "needs": needs,
        "search_history": needs
    }
