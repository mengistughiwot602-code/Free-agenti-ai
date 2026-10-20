import streamlit as st
import os
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool

st.set_page_config(page_title="Free Multi-AI Autonomous Researcher", layout="wide")
st.title("🤖 Free Multi-AI Autonomous Research Assistant")
st.write("Powered by GroqCloud, Google Gemini, OpenRouter, and Serper.dev!")

with st.sidebar:
    st.header("Free API Keys Setup")
    groq_key = st.text_input("Groq API Key", type="password")
    gemini_key = st.text_input("Google Gemini API Key", type="password")
    openrouter_key = st.text_input("OpenRouter API Key", type="password")
    serper_key = st.text_input("Serper API Key", type="password")
    st.markdown("---")
    st.markdown("Built with [CrewAI](https://crewai.com)")

topic = st.text_input("Enter your research topic:", value="")

if st.button("Start Free Autonomous Research"):
    if not all([groq_key, gemini_key, openrouter_key, serper_key]):
        st.error("Please provide all required free API keys in the sidebar.")
    elif not topic.strip():
        st.warning("Please enter a valid research topic.")
    else:
        os.environ["GROQ_API_KEY"] = groq_key
        os.environ["GEMINI_API_KEY"] = gemini_key
        os.environ["OPENROUTER_API_KEY"] = openrouter_key
        os.environ["SERPER_API_KEY"] = serper_key

        with st.status("Free multi-agent crew is working...", expanded=True) as status:
            st.write("Initializing search tools...")
            search_tool = SerperDevTool()

            st.write("Configuring specialized free model agents...")
            orchestrator = Agent(
                role="Orchestrator", goal=f"Plan research on {topic}",
                llm="groq/llama-3.3-70b-versatile", verbose=True
            )
            reviewer = Agent(
                role="Literature Reviewer", goal=f"Gather data on {topic}",
                tools=[search_tool], llm="gemini/gemini-1.5-flash", verbose=True
            )
            writer = Agent(
                role="Writer", goal=f"Synthesize report on {topic}",
                llm="openrouter/meta-llama/llama-3-70b-instruct", verbose=True
            )

            st.write("Assembling tasks and executing workflow...")
            t1 = Task(description=f"Plan research for {topic}", expected_output="Plan outline", agent=orchestrator)
            t2 = Task(description=f"Search facts for {topic}", expected_output="Fact list", agent=reviewer)
            t3 = Task(description=f"Draft report on {topic}", expected_output="Markdown report", agent=writer)

            crew = Crew(agents=[orchestrator, reviewer, writer], tasks=[t1, t2, t3], process=Process.sequential)
            result = crew.kickoff(inputs={"topic": topic})

            status.update(label="Research Completed successfully!", state="complete")

        st.subheader("Final Generated Report")
        st.markdown(result)
