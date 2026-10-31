import os
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import tool
from dotenv import load_dotenv

load_dotenv()

# LLM Setup - Uses GEMINI_API_KEY from .env
llm = LLM(
    model="gemini/gemini-2.0-flash", 
    temperature=0.1,
    max_retries=5
)

# -------------------------------------------------------------
# 2. CUSTOM TOOLS FOR AGENTS
# -------------------------------------------------------------
@tool("Web Scraper Tool")
def web_scraper_tool(industry: str, location: str) -> str:
    """Scrapes the web to find business leads based on industry and location."""
    print(f"\n[Tool] Scraping leads for {industry} in {location}...")
    
    dummy_data = f"""
    Found 2 High-Quality Leads for {industry} in {location}:
    1. Company: TechPulse Pakistan
       Contact Person: Kamran Ahmed (CEO)
       Email: kamran@techpulse.pk
       Need: Wants to automate customer support using AI.
       
    2. Company: Nexus Retailers
       Contact Person: Zainab Bibi (Operations Manager)
       Email: contact@nexusretail.com.pk
       Need: Looking for an AI lead generation system.
    """
    return dummy_data

# -------------------------------------------------------------
# 3. CREWAI AGENTS & TASKS
# -------------------------------------------------------------
lead_researcher = Agent(
    role="Lead Researcher",
    goal="Find high-quality B2B leads and their contact information.",
    backstory="You are an expert at deep-diving into the web, finding business emails, and analyzing what a company needs.",
    tools=[web_scraper_tool],
    llm=llm,
    verbose=True
)

email_writer = Agent(
    role="Email Copywriter",
    goal="Write highly personalized, high-converting cold emails.",
    backstory="You are a master of copywriting. You write short, professional, and compelling emails that get replies.",
    llm=llm,
    verbose=True
)

scrape_task = Task(
    description="Scrape leads for industry: {industry} in {location}. Use the provided tool to get the data.",
    expected_output="A structured list of companies, names, emails, and their specific business needs.",
    agent=lead_researcher
)

email_task = Task(
    description="Take the scraped leads and write a personalized pitch email for each lead offering AI Automation services.",
    expected_output="A set of custom emails ready to send, addressing each lead by their name and business problem.",
    agent=email_writer
)

lead_crew = Crew(
    agents=[lead_researcher, email_writer],
    tasks=[scrape_task, email_task],
    process=Process.sequential,
    verbose=True
)

# -------------------------------------------------------------
# 4. LANGGRAPH ORCHESTRATION
# -------------------------------------------------------------
class WorkflowState(dict):
    industry: str
    location: str
    result: str

def run_crewai_workflow(state: Dict[str, Any]) -> Dict[str, Any]:
    print("\n--- LangGraph NODE 1: Running CrewAI Multi-Agent Workflow ---")
    inputs = {
        "industry": state.get("industry", "AI Automation for Small Businesses"),
        "location": state.get("location", "Pakistan")
    }
    crew_output = lead_crew.kickoff(inputs=inputs)
    return {"result": str(crew_output)}

workflow = StateGraph(WorkflowState)
workflow.add_node("execute_crew", run_crewai_workflow)
workflow.set_entry_point("execute_crew")
workflow.add_edge("execute_crew", END)
app = workflow.compile()

if __name__ == "__main__":
    initial_input = {
        "industry": "AI Automation for Small Businesses",
        "location": "Pakistan"
    }
    print("### Autonomous Multi-Agent Workflow Started ###")
    final_state = app.invoke(initial_input)
    print("\n### FINAL WORKFLOW OUTPUT ###")
    print(final_state["result"])