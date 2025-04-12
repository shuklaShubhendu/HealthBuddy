import streamlit as st
from openai import OpenAI
import os

import json
import time
from operator import itemgetter
from typing import Dict, List

# Load environment variables
openai_api_key = st.secrets.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Doctor database with nutritionists
DOCTOR_DATABASE = [
    {"name": "Dr. John Smith", "specialty": "General Physician", "domain": "General Medicine", "rating": 4.8, "phone": "+1-555-123-4567", "email": "drjohn@example.com", "available": True},
    {"name": "Dr. Sarah Lee", "specialty": "Orthopedist", "domain": "Orthopedic Surgery", "rating": 4.9, "phone": "+1-555-987-6543", "email": "drsarah@example.com", "available": True},
    {"name": "Dr. Amit Patel", "specialty": "Orthopedist", "domain": "Orthopedic Surgery", "rating": 4.6, "phone": "+1-555-456-7890", "email": "dramit@example.com", "available": False},
    {"name": "Dr. Emily Chen", "specialty": "Pulmonologist", "domain": "Respiratory Medicine", "rating": 4.7, "phone": "+1-555-321-1234", "email": "dremily@example.com", "available": True},
    {"name": "Dr. Michael Brown", "specialty": "General Physician", "domain": "General Medicine", "rating": 4.5, "phone": "+1-555-654-3210", "email": "drmichael@example.com", "available": True},
    {"name": "Lisa Gupta, RDN", "specialty": "Nutritionist", "domain": "Dietetics", "rating": 4.9, "phone": "+1-555-111-2222", "email": "lisa.gupta@example.com", "available": True},
    {"name": "Priya Sharma, RDN", "specialty": "Nutritionist", "domain": "Vegetarian Nutrition", "rating": 4.7, "phone": "+1-555-333-4444", "email": "priya.sharma@example.com", "available": True},
    {"name": "Dr. Rachel Kim, RDN", "specialty": "Nutritionist", "domain": "Clinical Nutrition", "rating": 4.6, "phone": "+1-555-555-6666", "email": "rachel.kim@example.com", "available": False}
]

# Conversation log file
CONVERSATION_LOG = "conversation_log.json"

# Initialize conversation log
def init_conversation_log():
    if not os.path.exists(CONVERSATION_LOG):
        with open(CONVERSATION_LOG, 'w') as f:
            json.dump([], f)

# Save conversation to log
def save_conversation(user_input, bot_response, timestamp):
    conversation = {"timestamp": timestamp, "user_input": user_input, "bot_response": bot_response}
    with open(CONVERSATION_LOG, 'r') as f:
        logs = json.load(f)
    logs.append(conversation)
    with open(CONVERSATION_LOG, 'w') as f:
        json.dump(logs, f, indent=2)

# Tool: List available doctor specialties
def get_doctor_specialties() -> List[str]:
    return sorted(set(doc["specialty"] for doc in DOCTOR_DATABASE))

# Tool: Get highest-rated doctor for a specialty
def get_doctor_details(specialty: str) -> Dict:
    available_doctors = [doc for doc in DOCTOR_DATABASE if doc["specialty"].lower() == specialty.lower() and doc["available"]]
    if not available_doctors:
        return {"error": f"No available {specialty} found."}
    top_doctor = max(available_doctors, key=itemgetter("rating"))
    return {
        "name": top_doctor["name"],
        "specialty": top_doctor["specialty"],
        "phone": top_doctor["phone"],
        "email": top_doctor["email"],
        "rating": top_doctor["rating"]
    }

# Tools definition for OpenAI
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_doctor_specialties",
            "description": "Retrieve a list of available doctor specialties in the database.",
            "parameters": {}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_doctor_details",
            "description": "Get details of the highest-rated available doctor for a given specialty.",
            "parameters": {
                "type": "object",
                "properties": {
                    "specialty": {
                        "type": "string",
                        "description": "The specialty of the doctor (e.g., Nutritionist, Orthopedist)."
                    }
                },
                "required": ["specialty"]
            }
        }
    }
]

# Execute tool based on tool call
def execute_tool(tool_call):
    function_name = tool_call.function.name
    if function_name == "get_doctor_specialties":
        return get_doctor_specialties()
    elif function_name == "get_doctor_details":
        arguments = json.loads(tool_call.function.arguments)
        return get_doctor_details(arguments["specialty"])
    return None

# Get OpenAI response with tool calling
def get_openai_response(messages):
    try:
        # First attempt: Check if tools are needed
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
            max_tokens=200
        )
        response_message = response.choices[0].message

        # If tool calls are present, execute them
        if response_message.tool_calls:
            tool_results = []
            for tool_call in response_message.tool_calls:
                result = execute_tool(tool_call)
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result)
                })
            # Append tool results to messages and get final response
            messages.extend([response_message] + tool_results)
            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=200
            )
            return final_response.choices[0].message.content.strip()
        return response_message.content.strip()
    except Exception as e:
        return f"Sorry, something went wrong: {str(e)}"

# Streamlit UI and chatbot logic
def main():
    # Professional styling
 
    st.markdown('<div class="title">HealthBuddy</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your trusted companion for health and wellness</div>', unsafe_allow_html=True)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm here to support you with any health concerns. What's on your mind today?"}
        ]
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = [
            {
                "role": "system",
                "content": (
                    "You are HealthBuddy, a professional and empathetic chatbot designed to assist users with health and wellness concerns. "
                    "Engage in natural conversation, asking follow-up questions to understand symptoms or dietary issues. "
                    "For minor issues (e.g., mild tiredness), suggest simple remedies (e.g., rest, hydration) and continue the conversation. "
                    "Do NOT diagnose conditions. Use the 'get_doctor_specialties' tool to know available specialties when needed. "
                    "Only recommend a doctor after ~10 user messages, unless the user explicitly asks for one (e.g., 'I need a nutritionist'). "
                    "When recommending, use the 'get_doctor_details' tool to fetch details for the appropriate specialty (e.g., Nutritionist for dietary concerns). "
                    "For dietary issues (e.g., vegetarian diet, poor eating), consider a Nutritionist. For physical symptoms (e.g., cough, joint pain), consider other specialties. "
                    "Keep responses concise, professional, and supportive, avoiding medical jargon."
                )
            }
        ]
    if "user_message_count" not in st.session_state:
        st.session_state.user_message_count = 0

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_input = st.chat_input("Share how you're feeling or ask for a specialist...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        st.session_state.user_message_count += 1
        with st.chat_message("user"):
            st.markdown(user_input)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Check for direct doctor request
        if any(word in user_input.lower() for word in ["need", "want", "see", "consult", "nutritionist", "doctor"]):
            response = get_openai_response(st.session_state.conversation_history)
        else:
            # After 10 messages, allow doctor recommendation if needed
            if st.session_state.user_message_count >= 10:
                response = get_openai_response(st.session_state.conversation_history)
            else:
                response = get_openai_response(st.session_state.conversation_history)

        # Add bot response
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.conversation_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save to log
        init_conversation_log()
        save_conversation(user_input, response, timestamp)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Error: Please set your OpenAI API key in the .env file.")
    else:
        main()