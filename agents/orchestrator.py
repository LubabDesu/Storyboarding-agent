from agents.PlanningAgent import PlanningAgent
from agents.BaseAgent import AgentState

def orchestrator(model, tokenizer, state: dict) -> dict:
    
    print("Routed back to the Orchestrator Note")
    print("\nHere is the state I currently received as input!!\n")
    print(state)
    input = state["input"]
    knowledge = state.get("knowledge")
    # state_manager = AgentState(user_query=user_query)

    planning_agent = PlanningAgent(model=model, 
                                   tokenizer=tokenizer,
                                   state_manager=state)
    
    # You may want to add state_manager later if needed
    if knowledge :
        print("Knowledge exists\n")
        print(f"Hence the input is {input} and the knowledge we have is {knowledge}")
    else : 
        print("Knowledge doesn't exist\n")
        print(f"Hence the input is {input}")
        
    decision = planning_agent.devise_plan(input,knowledge, state['search_history'])
    
    print(f"\nResult status is : {decision.status}\n")
    print(f"Rationale behind that is : {decision.rationale}\n")
    print(f"Needs  is : {decision.needs}\n")
    
    status = decision.status
    rationale = decision.rationale
    needs = decision.needs
    print(f"Current Search History is {state['search_history']}\n")
    

    return {
        "reason_for_retry": rationale,
        "status": status,
        "needs": needs,
        "search_history": needs
    }
