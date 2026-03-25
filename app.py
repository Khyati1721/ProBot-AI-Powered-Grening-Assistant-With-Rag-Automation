import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
import requests
from langchain.tools import Tool
import data
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from voice import transcribe,text_to_speech 
from webscrap import ws
import pandas as pd
from graphs import graph_generator
import re
from mailsender import send_email,create_event
# venv\Scripts\activate
load_dotenv()
groq_api_key=os.getenv("Groq_API_KEY")
WETHER_API_KEY = os.getenv("WETHER_API_KEY")
# Initialize Groq LLM
llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.3-70b-versatile",
)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful gardening assistant.
Use ONLY the relevant information from the following context to answer the user's question.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""
)
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=data.vector_store.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": custom_prompt},
        return_source_documents=True
        )

# -----------------------------------------------------------------------------------------------

def get_weather(city: str):
    # Use your preferred weather API
    url = f"https://api.weatherapi.com/v1/current.json?key={WETHER_API_KEY}&q={city}"
    response = requests.get(url)
    data = response.json()
    return f"The current temperature in {city} is {data['current']['temp_c']}°C with {data['current']['condition']['text']}. and editional information are:{data}"
def rag_tool_func(query: str):
    output = st.session_state.rag_chain({"query": query})
    return output["result"]

def answer_question(question):
    search = DuckDuckGoSearchRun()
    return search.invoke(question)

def summarize_text(text):
    return llm.predict(f"summerise this:{text}")

def send_email_input_string(input_string: str):
    try:
        # Regex for each part with non-greedy matching
        to_match = re.search(r"to:\s*(.*?)(?=, subject:|$)", input_string, re.IGNORECASE)
        subject_match = re.search(r"subject:\s*(.*?)(?=, message:|$)", input_string, re.IGNORECASE)
        message_match = re.search(r"message:\s*(.*)", input_string, re.IGNORECASE | re.DOTALL)

        to = to_match.group(1).strip() if to_match else None
        subject = subject_match.group(1).strip() if subject_match else None
        message = message_match.group(1).strip() if message_match else None

        if to and subject and message:
            return send_email(to, subject, message)
        else:
            return "Error: Missing to, subject, or message."
    except Exception as e:
        return f"Error parsing input: {str(e)}"


def schedule_meeting_input_string(input_string: str):
    """
    Example input_string:
    'title: Gardening Talk, description: Discuss soil health, start_time: 2025-04-10T15:30:00, duration: 45'
    """
    try:
        parts = input_string.split(",")
        title = parts[0].split("title:")[1].strip()
        description = parts[1].split("description:")[1].strip()
        start_time = parts[2].split("start_time:")[1].strip()
        duration = int(parts[3].split("duration:")[1].strip().replace("'", ""))
        return create_event(title, description, start_time, duration)
    except Exception as e:
        return f"Error parsing meeting input: {str(e)}"


# -----------------------------------------------------------------------------------------------

meeting_tool = Tool(
    name="MeetingScheduler",
    func=schedule_meeting_input_string,
    description=(
        "Schedules a gardening meeting. Input format: "
        "'title: Gardening Talk, description: Discuss soil health, start_time: 2025-04-10T15:30:00, duration: 45'"
    )
)
weather_tool = Tool(
    name="WeatherFetcher",
    func=get_weather,
    description="Fetches current weather information for a given city."
)
rag_tool = Tool(
    name="GardeningRAGTool",
    func=rag_tool_func,
    description="""Use this tool to answer gardening questions using expert documents. """
)
question_tool = Tool(
    name="QA_Tool",
    func=answer_question,
    description="Answers gardening questions if not present in rag_tool."
)
summarization_tool = Tool(
    name="SummarizationTool",
    func=summarize_text,
    description="Summarizes long texts into concise points."
)
WebScraper_tool = Tool(
    name="WebScraperTool",
    func=ws,
    description="Use this tool when the user wants to buy or know the price of a product. It scrapes Google Shopping and returns a list of product names, prices, and buy links that can be shared with the user."
)
email_tool = Tool(
    name="EmailSender",
    func=send_email_input_string,
    description="Send an email. Format: 'to: <email>, subject: <subject>, message: <message>'."
)

# -----------------------------------------------------------------------------------------------

tools = [weather_tool,rag_tool,summarization_tool,question_tool,WebScraper_tool,email_tool,meeting_tool]  
if "agent" not in st.session_state:
    custome_prompt="""
Help user with gardening advice, plant care, pest control, composting, and related queries.
Be friendly, concise, and informative.

If RAGTool responds with "I don't know", then call QA_Tool to try answering the question.


Once you have the email, use the EmailSender tool to send the message.

IMPORTANT: If writing a mail never include my name section there.
When calling EmailSender, pass parameters in this format:
{{
    "to": "recipient@example.com",
    "subject": "Short subject here",
    "message": "Full message here"
}}

Example: If the user says, "Send an email to rohan.grow@gmail.com reminding him about pest control this weekend",
then call the tool like this:
{{
    "to": "rohan.grow@gmail.com",
    "subject": "Reminder about pest control",
    "message": "Just a reminder about pest control this weekend. Let me know if you have any questions."
}}
        """
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    st.session_state.agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True
    )

# streamlit UI code

st.set_page_config(page_title="ProBot",  layout="wide")
st.title("ProBot")
# creating messages list
if "messages" not in st.session_state:
    st.session_state.messages = []
# showing messages history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
            if msg.get("type") == "image":
                st.image(msg["content"], caption=msg["content"])
            else:
                st.markdown(msg["content"])
# user input
user_select= st.sidebar.selectbox(
        "How would you like to Communticate?",
        ("Text", "Voice")
    )


st.sidebar.subheader("Upload for Visualization")
uploaded_file=st.sidebar.file_uploader("upload your csv or excel file",type=["csv","xlsx"])
if uploaded_file and st.sidebar.button("Generate Charts"):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    if not df.empty:
        chart_path = graph_generator(df.head(30))
        if not chart_path:
            st.write("No charts were generated")
        else:
            for path in chart_path:
                st.image(path,caption=path)
                st.session_state.messages.append({"role": "assistant","type":"image","content":path})


if user_select == "Text":
    if prompt := st.chat_input("Ask anyting......"):
        st.session_state.messages.append({"role": "user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant","content":response})

else: 
    uploaded_file = st.audio_input("Record a Voice Message")
    if uploaded_file and st.button('🎤 Confirm Audio'):
        user_input = transcribe(uploaded_file)
        if user_input:
            # Show user message
            st.chat_message("user").markdown(user_input)
            st.session_state.messages .append({"role": "user", "content": user_input})

            # Run agent
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.agent.run(user_input)
                    if response.strip():
                        audio_file = text_to_speech(response)
                        st.sidebar.audio(audio_file, format='audio/mp3', autoplay=True)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
