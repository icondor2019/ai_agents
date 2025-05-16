from typing import TypedDict, Annotated, List
from langchain.tools import Tool
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI

from app.configuration.settings import settings
from app.utils.agent_hf_tools import get_hub_stats, get_weather_info


class AlfredAgent:
    class AgentState(TypedDict):
        messages: Annotated[List[AnyMessage], add_messages]

    def __init__(self):
        # Define tools
        self.hub_stats_tool = Tool(
            name="get_hub_stats",
            func=get_hub_stats,
            description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
        )

        self.weather_info_tool = Tool(
            name="get_weather_info",
            func=get_weather_info,
            description="Fetches dummy weather information for a given location."
        )

        # self.search_tool = DuckDuckGoSearchRun()

        # Initialize LLM with tools
        # llm = HuggingFaceEndpoint(
        #     repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        #     huggingfacehub_api_token=settings.HUGGINGFACEHUB_API_TOKEN,
        # )
        llm = ChatOpenAI(
            model_name="deepseek-r1-distill-llama-70b",  # TODO test with "llama3-8b-8192",
            openai_api_key=settings.GROQ_API_KEY,
            openai_api_base="https://api.groq.com/openai/v1",  # Groq's endpoint
            temperature=0.7
        )

        # chat = ChatHuggingFace(llm=llm, verbose=True)
        self.tools = [self.weather_info_tool, self.hub_stats_tool]
        self.chat_with_tools = llm.bind_tools(self.tools)

        # Build agent graph
        self.graph = self._build_graph()

    def _assistant_node(self, state: AgentState):
        return {
            "messages": [self.chat_with_tools.invoke(state["messages"])]
        }

    def _build_graph(self):
        builder = StateGraph(self.AgentState)
        builder.add_node("assistant", self._assistant_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")

        return builder.compile()

    def ask(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.graph.invoke({"messages": messages})
        return response["messages"][-1].content

    def ask_with_context(self, prompt: str, context: str) -> str:
        messages = [HumanMessage(content=context), HumanMessage(content=prompt)]
        response = self.graph.invoke({"messages": messages})
        return response["messages"][-1].content
