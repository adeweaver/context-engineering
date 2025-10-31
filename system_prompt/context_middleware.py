from dataclasses import dataclass
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    # RUNTIME CONTEXT: User and environment data available for this request
    user_id = request.runtime.context.user_id
    
    message_count = len(request.messages)
    
    print(f" Context Middleware: user_id={user_id}, messages={message_count}")
    
    prompt_parts = ["You are a helpful assistant."]
    
    # STATE CONTEXT: Conversation-based behavior
    if message_count > 10:
        prompt_parts.append("This is a long conversation - be extra concise.")
        print(" State Context: Long conversation detected")
    
    # STORE CONTEXT: User-based behavior
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)
    
    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        prompt_parts.append(f"User prefers {style} responses.")
        print(f"Store Context: Applied {style} preference")
    else:
        print("Store Context: No user preferences found")
    
    # Combine all context
    final_prompt = "\n".join(prompt_parts)
    print(f"Final prompt: {final_prompt}")
    
    return final_prompt

