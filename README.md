# AI Agent with LangGraph and FastAPI

## Overview
This project implements a sophisticated AI agent system using LangGraph's structured approach to agent development. The system is built with production-grade architecture using FastAPI and supports multiple LLM providers including GROQ and OpenAI.

## Features

### AI Agent Architecture
- Built using LangGraph's StateGraph for robust agent state management
- Modular tool integration system
- Support for contextual conversations
- Structured message handling with TypedDict

### LLM Integration
- Primary integration with GROQ's LLM API
- Compatible with OpenAI's API
- Flexible model selection (currently using llama3-8b-8192)
- Configurable temperature and response parameters

### Tools and Capabilities
- Hugging Face Hub statistics retrieval
- Weather information service
- Extensible tool architecture for easy additions

### Production-Ready Features
- FastAPI-based REST API
- Environment-based configuration
- Structured project layout
- Error handling and logging
- Type hints and annotations

### API Endpoints
- `POST /agent/ask`: Send queries to the AI agent
- `POST /agent/ask-with-context`: Send queries with additional context

## Project Structure
```
app/
├── agents/
│   └── agent_hf.py       # AI agent implementation
├── utils/
│   └── agent_hf_tools.py # Tool implementations
├── configuration/
│   └── settings.py       # Configuration management
├── api/
│   └── routes/          # API endpoints
└── main.py              # FastAPI application
```
