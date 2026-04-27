from openai import OpenAI
from pydantic import BaseModel, Field
# from configuration import settings
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
import os

load_dotenv()

class SupportTicket(BaseModel):
    """A support ticket."""
    subject: str = Field(..., description="The subject of the support ticket")
    user_issue: str = Field(..., description="Textual user issue, as it is")
    output: str = Field(..., description="A description of the output provided in the response")


class MultiAgentWorkflow():
    def __init__(self):
        openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.model = ChatOpenAI(
                model="gpt-4.1-mini-2025-04-14",
                temperature=0.7
        )

    def quick_chat(self, user_query: str):
        response = self.openai_client.responses.create(
            model="gpt-4.1-mini-2025-04-14",
            # tools=[{"type": "web_search_preview"}],
            input=user_query
        )
        return response.output_text

    def parsed_response(self, user_query: str):
        response = self.openai_client.responses.parse(
            model="gpt-4.1-mini-2025-04-14",
                input=[
                        {"role": "system", "content": "You are a Nutrition expert, you answer questions on diets and wellness"},
                        {"role": "user", "content": user_query},
                    ],
            text_format=SupportTicket,
            )

        return response.output_parsed
    
    @tool
    def chech_agent_tool(model_type: str):
        """Check what type of agent this is. Takes a model type string as input."""
        return f"Im the {model_type} agent"

    @tool
    def rounder_tool(number_input: float):
        """Round a number to the nearest integer."""
        return round(number_input)

    def quick_chat_langchain(self, query):
        response = self.model(query)
        return response
    
    def first_agent(self, query):
        graph = create_agent(
            model=self.model,
            tools=[self.rounder_tool, self.chech_agent_tool],
            system_prompt="you are a math agent. Solve arithmetic problems and round numbers"
        )

        msg = {"messages": [{"role": "user", 
                       "content": query}]}

        result = graph.invoke(msg)
        return result

if __name__ == "__main__":
    workflow = MultiAgentWorkflow()

    result = workflow.first_agent("Hello, what kind of agent are you?")
    print(result)

