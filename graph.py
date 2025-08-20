'''
Defines the graph of the agentic workflow 
'''

from IPython.display import Image, display
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal

class GraphState(TypedDict):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str = ""
    knowledge: List[Dict] = Field(default_factory=list)
    items: List[Dict] = Field(default_factory=list)
    current_task_index: Optional[int] = None
    needs_refinement: bool = False
    refinement_feedback: Optional[str] = None
    final_answer: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    attempt: int = 0                                    # how many times we've gathered
    reason_for_retry: Optional[str] = ""               # why orchestrator sent us back
    needs: Optional[List[str]] = None                # structured directives of what to fetch next
    search_history: Optional[List[str]] = None 
    status: Literal[
        "needs_info", 
        "ready_to_design", 
        "reflect_needed", 
        "refine_required", 
        "satisfactory", 
        "complete"
    ]
    
    
# ---- Import Node Functions (to be implemented in agents/*.py) ---- #
from agents.orchestrator import orchestrator
from agents.information_gatherer import information_gatherer
from agents.story_designer import story_designer
from agents.reflector import reflector
from agents.final_formatter import final_formatter

# workflow = StateGraph(GraphState)

# # Defining each nodes
# workflow.add_node("orchestrator", orchestrator)
# workflow.add_node("information_gatherer", information_gatherer)
# workflow.add_node("story_designer", story_designer)
# workflow.add_node("reflector", reflector)
# workflow.add_node("final_formatter", final_formatter)

# #Defining the edges 

# # 1. Orchestrator and Information Gatherer : Bi-directional
# workflow.add_conditional_edges(
#     "orchestrator",
#     lambda state: state["status"],
#     {
#         "needs_info": "information_gatherer",
#         "ready_to_design": "story_designer"
#     }
# )

# workflow.add_edge("information_gatherer", "orchestrator")

# # 2. Story Designer → Reflector
# workflow.add_edge("story_designer", "reflector")

# # 3. Reflector Conditional Edges
# workflow.add_conditional_edges(
#     "reflector",
#     lambda state: state["status"],
#     {
#         "satisfactory": "final_formatter",
#         "refine_required": "orchestrator",
#         "needs_info": "information_gatherer"
#     }
# )

# # 4. Final Formatter → End
# workflow.add_conditional_edges("final_formatter", 
#         lambda state: state["status"],
#     {
#         "satisfactory": END
#     })

# # ---- Set Entry Point ---- #
# workflow.set_entry_point("orchestrator")

# # ---- Compile the Graph ---- #
# graph = workflow.compile()


# graph.py

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import Optional, Literal

# Define the shared state between nodes
class GraphState(TypedDict):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str = ""
    knowledge: List[Dict] = Field(default_factory=list)
    items: List[Dict] = Field(default_factory=list)
    current_task_index: Optional[int] = None
    needs_refinement: bool = False
    refinement_feedback: Optional[str] = None
    final_answer: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    attempt: int = 0                                    # how many times we've gathered
    reason_for_retry: Optional[str] = ""               # why orchestrator sent us back
    needs: Optional[List[str]] = None                # structured directives of what to fetch next
    search_history: Optional[List[str]] = None 
    status: Literal[
        "needs_info", 
        "ready_to_design", 
        "reflect_needed", 
        "refine_required", 
        "satisfactory", 
        "complete"
    ]

# ---- Graph Builder Function ---- #

def build_graph(
    orchestrator_node,
    information_gatherer_node,
    final_formatter_node
):
    wf = StateGraph(GraphState)

    wf.add_node("orchestrator", orchestrator_node)
    wf.add_node("information_gatherer", information_gatherer_node)
    wf.add_node("final_formatter", final_formatter_node)

    # orchestrator routes directly to final_formatter when ready
    wf.add_conditional_edges(
        "orchestrator",
        lambda s: s["status"],
        {
            "needs_info": "information_gatherer",
            "ready_to_design": "final_formatter",
        },
    )

    # gatherer always returns to orchestrator
    wf.add_edge("information_gatherer", "orchestrator")

    # final output ends
    wf.add_edge("final_formatter", END)

    wf.set_entry_point("orchestrator")
    return wf.compile()


# Testing the workflow of orch and info gatherer --
# --- Test Graph for Orchestrator Loop ---
def create_test_workflow(orchestrator_node, gatherer_node):
    """Builds the isolated loop for testing."""
    test_workflow = StateGraph(GraphState)
    test_workflow.add_node("orchestrator", orchestrator_node)
    test_workflow.add_node("information_gatherer", gatherer_node)

    test_workflow.set_entry_point("orchestrator")
    test_workflow.add_conditional_edges(
        "orchestrator",
        lambda state: state["status"],
        {"needs_info": "information_gatherer", "ready_to_design": END}
    )
    test_workflow.add_edge("information_gatherer", "orchestrator")
    return test_workflow.compile()