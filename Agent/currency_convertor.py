import os
from dotenv import load_dotenv
from ollama import chat
import requests
from langchain_core.messages import HumanMessage

load_dotenv()

def main():
    API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')

    def get_conversion_rate(base_currency: str, target_currency: str) -> float:
        """Fetches the current exchange rate from the API."""
        API_KEY = os.getenv('EXCHANGE_RATE_API_KEY')
        endpoint = f'https://v6.exchangerate-api.com/v6/{API_KEY}/pair/{base_currency}/{target_currency}'
        response = requests.get(endpoint)
        return response.json().get('conversion_rate', 0.0)
    
    def calculate_conversion(amount: float, rate: float) -> float:
        """Multiplies the amount by the conversion rate."""
        return round(amount * rate, 2)

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'get_conversion_rate',
                'description': 'Get the exchange rate between two currencies',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'base_currency': {'type': 'string'},
                        'target_currency': {'type': 'string'},
                    },
                    'required': ['base_currency', 'target_currency'],
                },
            },
        },
        {
            'type': 'function',
            'function': {
                'name': 'calculate_conversion',
                'description': 'Multiply an amount by a conversion rate',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'amount': {'type': 'number'},
                        'rate': {'type': 'number'},
                    },
                    'required': ['amount', 'rate'],
                },
            },
        }
    ]

    messages = [{'role': 'user', 'content': HumanMessage(content='Can you convert 50 UAH to USD?').content}]

    available_functions = {
        'get_conversion_rate': get_conversion_rate,
        'calculate_conversion': calculate_conversion,
    }

    while True:
        response = chat(model='llama3.1', messages=messages, tools=tools)
            
        if not response['message'].get('tool_calls'):
            print(f"\nAssistant: {response['message']['content']}")
            break

        messages.append(response['message'])

        for call in response['message']['tool_calls']:
            func_name = call['function']['name']
            func_args = call['function']['arguments']
            
            print(f"--> Calling tool: {func_name}({func_args})")
        
            func_to_run = available_functions[func_name]
            result = func_to_run(**func_args)

            messages.append({
                'role': 'tool',
                'content': str(result),
            })


if __name__ == "__main__":
    main()