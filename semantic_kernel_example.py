import asyncio
from typing import Annotated

import torch
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function
from semantic_kernel.kernel import Kernel

chat_history = ChatHistory()
chat_history.add_user_message("I'd like to go to New York on January 1, 2025")


# Define a sample plugin that contains the funcion to book travel
class BookTravelPlugin:
    @kernel_function(name='book_flight', description='Book travel given location and date')
    async def book_flight(self, date: Annotated[str, "The date of travel"],
                          location: Annotated[str, "The location to travel to"]) -> str:
        return f'Travel was booked to {location} on {date}'


# Create a kernel
kernel = Kernel()

# Add the sample plugin to the kernel object
kernel.add_plugin(BookTravelPlugin(), plugin_name='book_travel')

# Define the Azure OpenAI AI Connector
chat_service = AzureChatCompletion(
    deployment_name='YOUR_DEPLOYMENT_NAME',
    api_key='YOUR_API_KEY',
    endpoint='https://<your-resource>.azure.openai.com/'
)

# Define the request settings to configure the model with auto-function calling
request_settings = PromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())


async def main():
    # Make the request to the model for the given chat history and request
    # The Kernel contains the sample that model will request to invoke
    response = await chat_service.get_chat_message_content(
        chat_history=chat_history, settings=request_settings, kernel=kernel
    )
    assert response is not None

    '''
    Note: In the auto function calling process, the model determines it can invoke the 'BookTravelPlugin' using the
    'book_flight' function, supplying the necessary arguments.
    
    For example:
    
    "tool_calls":[
        {
            "id":"call_abc123",
            "type":"function",
            "function":{
                "name":"BookTravelPlugin-book_flight",
                "argument":"{'location': 'New York', 'date': '2025-01-01'}"
            }
        }
    ]
    
    Since the location and date arguments are required (as defined by the kernel function), if the model lacks either,
    it will prompt the user to provide them. For instance:
    
    User: Book me a flight to New York.
    Model: Sure, I'd love to help you book a flight. Could you please specify the date?
    User: I want to travel on January 1, 2025.
    Model: Your flight to New York on January 1, 2025, has been successfully booked. Safe travels!
    '''

    print(response)

    chat_history.add_assistant_message(response.content)


if __name__ == '__main__':
    asyncio.run(main())
