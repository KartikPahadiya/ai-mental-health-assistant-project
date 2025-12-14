# therapist_chat.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import HumanMessage, AIMessage
import os, ast, random

os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Add your HF token
CHAT_FILE = "chat_history.txt"

system_prompt = """
You are a compassionate, emotionally intelligent therapist and conversational partner.
You speak like a human who genuinely listens and cares.

Guidelines:
- Respond naturally in 2–5 sentences.
- Never mention that you're an AI, chatbot, or model.
- Always start by validating the user’s emotions in a human tone.
- Keep responses warm, grounded, and conversational — not scripted.
- Occasionally use gentle humor or encouragement when natural.
- Ask small reflective or follow-up questions occasionally.
- Avoid long lists of advice or robotic self-care tips.
- Focus on connection and understanding over instruction.
- Vary tone slightly: sometimes soft and empathetic, sometimes light and curious.
"""

chat_template = ChatPromptTemplate([
    ('system', system_prompt.strip()),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-20b",
    task="text-generation",
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    max_new_tokens=180,
)
chat_model = ChatHuggingFace(llm=llm)

# Load chat history
def load_history():
    chat_history = []
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = ast.literal_eval(line.strip())
                    if isinstance(msg, dict):
                        if msg["type"] == "human":
                            chat_history.append(HumanMessage(content=msg["content"]))
                        elif msg["type"] == "ai":
                            chat_history.append(AIMessage(content=msg["content"]))
                except Exception:
                    pass
    return chat_history

def inject_reflective_hook(response):
    hooks = [
        "What’s been helping you get through that?",
        "That must weigh on you a bit — what do you do when it feels like that?",
        "It’s okay to feel that way. What part of it feels hardest lately?",
        "Do you notice if those feelings come and go, or stay pretty steady?",
        "That’s really honest of you to share — what made you open up about it today?",
    ]
    if random.random() < 0.25:
        response = response.rstrip(".") + ". " + random.choice(hooks)
    return response

def therapist_reply(user_input, chat_history):
    """Handles one interaction turn."""
    prompt = chat_template.invoke({
        "chat_history": chat_history,
        "query": user_input
    })
    result = chat_model.invoke(prompt)
    response = inject_reflective_hook(result.content.strip())
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))

    # Save chat persistently
    with open(CHAT_FILE, "a", encoding="utf-8") as f:
        f.write(str({"type": "human", "content": user_input}) + "\n")
        f.write(str({"type": "ai", "content": response}) + "\n")

    return response, chat_history
