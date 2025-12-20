from langchain_community.llms import Ollama


def main():
    m = Ollama(model="tinyllama")

    response = m.invoke("When did Ukraine declare independence?")
    print(response)

if __name__ == "__main__":
    main()