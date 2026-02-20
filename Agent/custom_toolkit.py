from langchain_core.tools import tool


def main():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b
    
    @tool
    def multiply(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b
    
    class MathToolKit:
        def get_tools(self):
            return [add, multiply]
    
    toolkit = MathToolKit()
    tools = toolkit.get_tools()

    for t in tools:
        print(t.name, "->", t.description)

if __name__ == "__main__":
    main()