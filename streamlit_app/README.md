# 🏨 Motel One Strategic Repositioning Agent System

A multi-agent strategic analysis system using LangGraph's Supervisor Architecture with the Question Decomposition Protocol.

## Overview

This application uses a team of specialized AI agents to analyze strategic repositioning challenges:

- **🔍 Research Agent** - Gathers market data, trends, and competitor analysis using Tavily search
- **👥 Stakeholder Analyst** - Analyzes implicit needs and stakeholder perspectives
- **🎯 Strategy Agent** - Identifies tactical traps and strategic opportunities
- **📋 Supervisor** - Orchestrates the workflow and synthesizes final output

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

You have two options:

**Option A: Use a `.env` file (for command-line usage)**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Option B: Enter keys in the Streamlit sidebar (for web app)**
The app provides input fields for API keys in the sidebar.

### 3. Get API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Tavily API Key**: Get from [Tavily](https://tavily.com/)

## Running the Application

### Streamlit Web App

```bash
streamlit run app.py
```

This will open a web browser with the interactive application where you can:
- Enter your API keys in the sidebar
- Run the strategic analysis
- View results in organized tabs
- Ask follow-up questions

### Command Line

```bash
python main.py
```

This runs the analysis in the terminal with an interactive follow-up mode.

## Project Structure

```
streamlit_app/
├── app.py              # Streamlit web application
├── main.py             # Command-line script
├── agents.py           # Core agent logic and graph builder
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
└── README.md           # This file
```

## The Challenge

The system is designed to solve the **Motel One Intergenerational Brand Gap**:

- **Current State**: 90% Gen-X/Boomers, only 3% Gen-Z bookings
- **Target**: Capture Gen-Z travel spend (35% of market by 2027)
- **Revenue at Stake**: €400M by 2028
- **Budget**: €8M over 18 months
- **Constraint**: Cannot alienate existing Boomer/Gen-X base

## Agent Workflow

1. **Research Phase** - Market intelligence gathering
2. **Synthesis Phase** - Research data synthesis
3. **Stakeholder Analysis** - Phase 1 of Question Decomposition Protocol
4. **Strategy Generation** - Phases 2 & 3 of QDP
5. **Final Brief** - Executive-ready strategic recommendations

## License

MIT License
