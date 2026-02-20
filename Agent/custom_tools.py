from langchain_core.tools import tool

def main():

    # 1 - Create a function
    def multiply1(a, b):
        """Multiply two numbers"""
        return a * b

    # 2 - Adding type hints
    def multiply2(a: int, b: int) -> int:
        """Multiply two numbers"""
        return a * b
    
    # 3 - Adding tool decorator
    @tool
    def multiply3(a: int, b: int) -> int:
        """This tool takes 2 integers and multiplies them"""
        return a * b
    
    print(multiply1(5, 4))
    print(multiply2(5, 4))
    result = multiply3.invoke({"a": 5, "b": 4})
    print(result)
    print(multiply3.name)
    print(multiply3.description)
    print(multiply3.args)
    print(multiply3.args_schema.model_json_schema())

if __name__ == "__main__":
    main()