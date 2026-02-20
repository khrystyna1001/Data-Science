from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

def main():

    # DuckDuckGoSearchRun
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke("Top news today")
    print(results)
    print(search_tool.name)
    print(search_tool.description)
    print(search_tool.args)

    # ShellTool
    shell_tool = ShellTool()
    results = shell_tool.invoke("rmdir random")
    print(results)

if __name__ == "__main__":
    main()