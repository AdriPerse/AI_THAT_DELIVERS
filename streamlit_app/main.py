# Multi-Agent Strategic Analysis System for Motel One Repositioning
# Main script for running the analysis programmatically

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
import os

from agents import (
    AgentState,
    build_strategic_agent_graph,
    ask_strategist,
    CHALLENGE_BRIEF
)

# Load environment variables
load_dotenv()


def main():
    """Main function to run the strategic analysis"""
    print("🏨 Motel One Strategic Repositioning Agent System")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ Error: TAVILY_API_KEY not found in environment variables")
        return
    
    # Initialize LLMs
    llm = ChatOpenAI(temperature=0.3, model="gpt-4o-mini")
    strategic_llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")
    
    # Build the graph
    print("✅ Building strategic agent graph...")
    strategic_agent = build_strategic_agent_graph(llm, strategic_llm)
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=CHALLENGE_BRIEF)],
        "research_data": "",
        "stakeholder_analysis": "",
        "strategic_angles": "",
        "current_phase": "start",
        "next_agent": "research_agent"
    }
    
    print("\n🚀 Initiating Strategic Analysis...\n")
    print(CHALLENGE_BRIEF)
    print("=" * 60)
    
    # Run the agent with streaming to see progress
    for step in strategic_agent.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in step.items():
            print(f"\n📍 Agent: {node_name.upper()}")
            print("-" * 40)
            
            if "messages" in node_output and node_output["messages"]:
                last_msg = node_output["messages"][-1]
                if hasattr(last_msg, 'content') and last_msg.content:
                    content_preview = last_msg.content[:500] + "..." if len(last_msg.content) > 500 else last_msg.content
                    print(content_preview)
            
            if "current_phase" in node_output:
                print(f"\n✅ Phase: {node_output['current_phase']}")
    
    print("\n" + "=" * 60)
    print("🎯 Strategic Analysis Complete!")
    
    # Get the final state
    final_result = strategic_agent.invoke(initial_state)
    
    # Display final strategic brief
    print("\n" + "=" * 80)
    print("📋 FINAL STRATEGIC BRIEF - MOTEL ONE REPOSITIONING")
    print("=" * 80)
    print(final_result["messages"][-1].content)
    
    # Interactive follow-up
    print("\n" + "=" * 60)
    print("💬 Interactive Mode - Ask follow-up questions")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        if question:
            response = ask_strategist(question, final_result, llm)
            print(f"\n🎯 Strategist: {response}")


if __name__ == "__main__":
    main()
