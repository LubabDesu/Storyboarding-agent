from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uuid
import datetime

# Define enhanced state schema
class TaskTool(BaseModel):
    """Tool selected for a task."""
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)

class TaskExecution(BaseModel):
    """Record of a task execution."""
    attempt: int
    tool: Optional[TaskTool] = None
    result: Optional[str] = None
    timestamp: str

class Task(BaseModel):
    """A single task to be completed."""
    id: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    executions: List[TaskExecution] = Field(default_factory=list)
    reflection: Optional[str] = None
    
    def latest_result(self) -> Optional[str]:
        """Get the most recent execution result."""
        if not self.executions:
            return None
        return self.executions[-1].result

class AgentState(BaseModel):
    """The state of the agent workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    input: str = ""
    knowledge: List[Dict] = Field(default_factory=list)
    items: List[Dict] = Field(default_factory=list)
    tasks: List[Task] = Field(default_factory=list)
    current_task_index: Optional[int] = None
    needs_refinement: bool = False
    refinement_feedback: Optional[str] = None
    final_answer: Optional[str] = None
    conversation_history: List[Dict[str, Any]] = Field(default_factory=list)
    attempt: int = 0                                    # how many times we've gathered
    reason_for_retry: Optional[str] = ""               # why orchestrator sent us back
    needs: Optional[List[str]] = None                # structured directives of what to fetch next
    search_history: Optional[List[str]] = None          # queries attempted so far
    
    def current_task(self) -> Optional[Task]:
        """Get the current task being processed."""
        if self.current_task_index is not None and 0 <= self.current_task_index < len(self.tasks):
            return self.tasks[self.current_task_index]
        return None
    
    def all_tasks_completed(self) -> bool:
        """Check if all tasks are completed."""
        return all(task.status == "completed" for task in self.tasks)
    
    def get_task_summary(self) -> str:
        """Get a summary of all tasks and their status."""
        summary = []
        for i, task in enumerate(self.tasks):
            summary.append(f"Task {i+1}: {task.description} - Status: {task.status}")
            if task.executions:
                latest_exec = task.executions[-1]
                summary.append(f"  Latest execution result: {latest_exec.result}")
            if task.reflection:
                summary.append(f"  Reflection: {task.reflection}")
        return "\n".join(summary)
    
    def add_to_conversation(self, role: str, content: str):
        """Add a message to the conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        })


class BaseAgent(BaseModel):
    state_manager : AgentState 
    
    class Config : 
        arbitrary_types_allowed = True
        
    def __init__(self, state_manager=None, **data):
        # Combine positional and keyword arguments
        if state_manager is not None:
            data['state_manager'] = state_manager
        super().__init__(**data)

    def log(self, message):
        # Logging logic
        print(f"Log: {message}")
        self.state_manager.add_to_conversation("system", message)

    def handle_error(self, error):
        # Error handling logic
        self.log(f"Error: {error}")

    def get_state(self) -> AgentState:
        # State management logic to get the current state
        return self.state_manager

    def update_state(self, new_state):
        # State management logic to update the state
        return self.state_manager.update_state(new_state)

    def communicate(self, message, recipient):
        # Communication logic to send messages to other agents or users
        self.log(f"Message to {recipient}: {message}")

        

# <------------- Tests ------------>
'''

print("Running tests now in BaseAgent.py")
agent_state = AgentState(
    user_query="What is the weather today?"
)
agent = BaseAgent(agent_state)
assert agent.state_manager.next_step == "plan", "Initial next_step should be 'plan'"
agent.log("Test message")
assert agent.state_manager.conversation_history[-1]['content'] == "Test message", "Log message not recorded correctly"

'''
