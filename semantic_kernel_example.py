import asyncio
from typing import Annotated

from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings
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
kernel=Kernel()

# Add the sample plugin to the kernel object
kernel.add_plugin(BookTravelPlugin(),plugin_name='book_travel')

# Define the Azure OpenAI AI Connector
chat_service = AzureChatCompletion(
    deployment_name='YOUR_DEPLOYMENT_NAME',
    api_key='YOUR_API_KEY',
    endpoint='https://<your-resource>.azure.openai.com/'
)