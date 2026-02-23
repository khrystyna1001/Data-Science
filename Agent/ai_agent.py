from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import Ollama

def main():
    llm = Ollama(model="deepseek-r1")
    search_tool = DuckDuckGoSearchRun()

    prompt = hub.pull('hwchase17/react')

    agent = create_react_agent(
        llm = llm,
        tools = [search_tool],
        prompt = prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=[search_tool],
        verbose=True,
        handle_parsing_errors=True
    )
    response = agent_executor.invoke({"input": "Find the capital of Italy"})
    print(response)
    print(response['output'])


if __name__ == "__main__":
    main()