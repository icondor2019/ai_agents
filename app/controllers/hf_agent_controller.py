from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.agents.agent_hf import AlfredAgent

api = APIRouter()
# Instantiate the agent once
agent = AlfredAgent()


# Request model
class QueryRequest(BaseModel):
    prompt: str


# Response model
class QueryResponse(BaseModel):
    response: str


@api.post("/ask", response_model=QueryResponse)
def ask_agent(query: QueryRequest):
    try:
        response_text = agent.ask(query.prompt)
        return QueryResponse(response=response_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
