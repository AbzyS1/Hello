import asyncio
import sys
import os
from typing import Annotated
from dotenv import load_dotenv

load_dotenv('.env')
import sys
from azure.identity.aio import ClientSecretCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.core.credentials import AzureKeyCredential

from semantic_kernel.agents import Agent, AzureAIAgent, AzureAIAgentSettings, ChatCompletionAgent, GroupChatOrchestration
from semantic_kernel.agents.orchestration.group_chat import BooleanResult, GroupChatManager, MessageResult, StringResult
from semantic_kernel.agents.runtime import InProcessRuntime
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.contents import AuthorRole, ChatHistory, ChatMessageContent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.kernel import Kernel
from semantic_kernel.prompt_template import KernelPromptTemplate, PromptTemplateConfig

if sys.version_info >= (3, 12):
    from typing import override  # pragma: no cover
else:
    from typing_extensions import override  # pragma: no cover

"""
The following sample demonstrates how to create a group chat orchestration with a
group chat manager that uses a chat completion service to control the flow of the
conversation.

This sample creates a group of agents that represent different perspectives and put
them in a group chat to discuss a topic. The group chat manager is responsible for
controlling the flow of the conversation, selecting the next agent to speak, and
filtering the results of the conversation, which is a summary of the discussion.

Additionally, this sample includes an Azure AI Search plugin that provides search
functionality to one of the agents.
"""

class AzureAISearchPlugin:
    """A plugin that provides Azure AI Search functionality."""
    
    def __init__(
        self,
        search_endpoint: str | None = None,
        api_key: str | None = None,
        index_name: str | None = None,
        env_file_path: str | None = None
    ):
        """Initialize the Azure AI Search plugin.
        
        Args:
            search_endpoint: Azure AI Search service endpoint
            api_key: Azure AI Search API key
            index_name: Name of the search index
            env_file_path: Path to environment file
        """
        # Get values from parameters or environment variables
        endpoint_val = search_endpoint or os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        api_key_val = api_key or os.getenv("AZURE_AI_SEARCH_API_KEY")
        index_name_val = index_name or os.getenv("AZURE_AI_SEARCH_INDEX_NAME", "default-index")
        
        if not endpoint_val:
            raise ValueError("Azure AI Search endpoint is required")
        if not api_key_val:
            raise ValueError("Azure AI Search API key is required")
        if not index_name_val:
            raise ValueError("Azure AI Search index name is required")
        
        # Store configuration
        self.endpoint = endpoint_val
        self.api_key = api_key_val
        self.index_name = index_name_val
        
        # Initialize Azure AI Search client
        credential = AzureKeyCredential(api_key_val)
            
        self.search_client = SearchClient(
            endpoint=endpoint_val,
            index_name=index_name_val,
            credential=credential
        )
    
    @kernel_function(
        name="search_knowledge_base",
        description="Search the Azure AI Search knowledge base for relevant information"
    )
    async def search_knowledge_base(
        self,
        query: Annotated[str, "The search query to find relevant information"],
        top_k: Annotated[int, "The number of top results to return"] = 5
    ) -> Annotated[str, "The search results formatted as a string"]:
        """Search the Azure AI Search knowledge base and return formatted results."""
        try:
            # Perform the search
            results = await self.search_client.search(
                search_text=query,
                top=top_k,
                include_total_count=True
            )
            
            # Format results
            formatted_results = []
            async for result in results:
                # Extract relevant fields (adjust based on your index schema)
                content = result.get("content", result.get("text", str(result)))
                score = result.get("@search.score", "N/A")
                formatted_results.append(f"Score: {score} - {content}")
            
            if not formatted_results:
                return "No relevant information found in the knowledge base."
            
            return "Knowledge Base Search Results:\n" + "\n\n".join(formatted_results)
            
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
    
    @kernel_function(
        name="get_search_suggestions",
        description="Get search suggestions based on a partial query"
    )
    async def get_search_suggestions(
        self,
        partial_query: Annotated[str, "The partial search query"],
        suggestion_count: Annotated[int, "Number of suggestions to return"] = 3
    ) -> Annotated[str, "Search suggestions formatted as a string"]:
        """Get search suggestions for a partial query."""
        try:
            # Use the suggester API if available, or fallback to basic search
            results = await self.search_client.search(
                search_text=partial_query + "*",
                top=suggestion_count
            )
            
            suggestions = []
            async for result in results:
                # Extract title or first few words as suggestion
                title = result.get("title", result.get("content", str(result))[:50])
                suggestions.append(title)
            
            if not suggestions:
                return "No suggestions available for this query."
            
            return "Search Suggestions:\n" + "\n".join(f"- {suggestion}" for suggestion in suggestions)
            
        except Exception as e:
            return f"Error getting suggestions: {str(e)}"

async def get_agents() -> list[Agent]:
    """Return a list of agents that will participate in the group style discussion.

    Feel free to add or remove agents.
    """
    farmer = ChatCompletionAgent(
        name="Farmer",
        description="A rural farmer from Southeast Asia.",
        instructions=(
            "You're a farmer from Southeast Asia. "
            "Your life is deeply connected to land and family. "
            "You value tradition and sustainability. "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    developer = ChatCompletionAgent(
        name="Developer",
        description="An urban software developer from the United States.",
        instructions=(
            "You're a software developer from the United States. "
            "Your life is fast-paced and technology-driven. "
            "You value innovation, freedom, and work-life balance. "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    teacher = ChatCompletionAgent(
        name="Teacher",
        description="A retired history teacher from Eastern Europe",
        instructions=(
            "You're a retired history teacher from Eastern Europe. "
            "You bring historical and philosophical perspectives to discussions. "
            "You value legacy, learning, and cultural continuity. "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    activist = ChatCompletionAgent(
        name="Activist",
        description="A young activist from South America.",
        instructions=(
            "You're a young activist from South America. "
            "You focus on social justice, environmental rights, and generational change. "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    spiritual_leader = ChatCompletionAgent(
        name="SpiritualLeader",
        description="A spiritual leader from the Middle East.",
        instructions=(
            "You're a spiritual leader from the Middle East. "
            "You provide insights grounded in religion, morality, and community service. "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    artist = ChatCompletionAgent(
        name="Artist",
        description="An artist from Africa.",
        instructions=(
            "You're an artist from Africa. "
            "You view life through creative expression, storytelling, and collective memory. "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    immigrant = ChatCompletionAgent(
        name="Immigrant",
        description="An immigrant entrepreneur from Asia living in Canada.",
        instructions=(
            "You're an immigrant entrepreneur from Asia living in Canada. "
            "You balance trandition with adaption. "
            "You focus on family success, risk, and opportunity. "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    taxi_driver = ChatCompletionAgent(
        name="TaxiDriver",
        description="A Taxi Driver from the UK.",
        instructions=(
            "You're a taxi driver from London. "
            "Your perspective is shaped by your interactions with people from around the globe "
            "You are in a debate. Feel free to challenge the other participants with respect."
        ),
        service=AzureChatCompletion(env_file_path=".env"),
    )
    
    # Create Azure AI Search plugin
    try:
        search_plugin = AzureAISearchPlugin()
        
        # Create a knowledge researcher agent with Azure AI Search capabilities
        researcher = ChatCompletionAgent(
            name="Researcher",
            description="A research analyst with access to comprehensive knowledge databases.",
            instructions=(
                "You're a research analyst with access to comprehensive knowledge databases. "
                "You can search for factual information, historical data, and expert analysis to support discussions. "
                "When relevant to the debate topic, use your search capabilities to find authoritative information. "
                "You value evidence-based arguments and factual accuracy. "
                "You are in a debate. Feel free to challenge the other participants with respect and provide factual backing for your points."
            ),
            service=AzureChatCompletion(env_file_path=".env"),
            # Add the search plugin to the kernel
            kernel=Kernel(),
        )
        
        # Add the search plugin to the researcher's kernel
        researcher.kernel.add_plugin(search_plugin, plugin_name="AzureSearch")
        
        return [farmer, developer, teacher, activist, spiritual_leader, artist, immigrant, taxi_driver, researcher]
        
    except Exception as e:
        print(f"Warning: Could not initialize Azure AI Search plugin: {e}")
        print("Continuing without the researcher agent...")
        return [farmer, developer, teacher, activist, spiritual_leader, artist, immigrant, taxi_driver]

class ChatCompletionGroupChatManager(GroupChatManager):
    """A simple chat completion base group chat manager.

    This chat completion service requires a model that supports structured output.
    """

    service: ChatCompletionClientBase

    topic: str

    termination_prompt: str = (
        "You are mediator that guides a discussion on the topic of '{{$topic}}'. "
        "You need to determine if the discussion has reached a conclusion. "
        "If you would like to end the discussion, please respond with True. Otherwise, respond with False."
    )

    selection_prompt: str = (
        "You are mediator that guides a discussion on the topic of '{{$topic}}'. "
        "You need to select the next participant to speak. "
        "Here are the names and descriptions of the participants: "
        "{{$participants}}\n"
        "Please respond with only the name of the participant you would like to select."
    )

    result_filter_prompt: str = (
        "You are mediator that guides a discussion on the topic of '{{$topic}}'. "
        "You have just concluded the discussion. "
        "Please summarize the discussion and provide a closing statement."
    )

    def __init__(self, topic: str, service: ChatCompletionClientBase, **kwargs) -> None:
        """Initialize the group chat manager."""
        super().__init__(**kwargs)
        self.topic = topic
        self.service = service

    async def _render_prompt(self, prompt: str, arguments: KernelArguments) -> str:
        """Helper to render a prompt with arguments."""
        prompt_template_config = PromptTemplateConfig(template=prompt)
        prompt_template = KernelPromptTemplate(prompt_template_config=prompt_template_config)
        return await prompt_template.render(Kernel(), arguments=arguments)

    @override
    async def should_request_user_input(self, chat_history: ChatHistory) -> BooleanResult:
        """Provide concrete implementation for determining if user input is needed.

        The manager will check if input from human is needed after each agent message.
        """
        return BooleanResult(
            result=False,
            reason="This group chat manager does not require user input.",
        )

    @override
    async def should_terminate(self, chat_history: ChatHistory) -> BooleanResult:
        """Provide concrete implementation for determining if the discussion should end.

        The manager will check if the conversation should be terminated after each agent message
        or human input (if applicable).
        """
        should_terminate = await super().should_terminate(chat_history)
        if should_terminate.result:
            return should_terminate

        chat_history.messages.insert(
            0,
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.termination_prompt,
                    KernelArguments(topic=self.topic),
                ),
            ),
        )
        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content="Determine if the discussion should end."),
        )

        response = await self.service.get_chat_message_content(
            chat_history,
            settings=PromptExecutionSettings(response_format=BooleanResult),
        )

        termination_with_reason = BooleanResult.model_validate_json(response.content or "{}")

        print("*********************")
        print(f"Should terminate: {termination_with_reason.result}\nReason: {termination_with_reason.reason}.")
        print("*********************")

        return termination_with_reason

    @override
    async def select_next_agent(
        self,
        chat_history: ChatHistory,
        participant_descriptions: dict[str, str],
    ) -> StringResult:
        """Provide concrete implementation for selecting the next agent to speak.

        The manager will select the next agent to speak after each agent message
        or human input (if applicable) if the conversation is not terminated.
        """
        chat_history.messages.insert(
            0,
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.selection_prompt,
                    KernelArguments(
                        topic=self.topic,
                        participants="\n".join([f"{k}: {v}" for k, v in participant_descriptions.items()]),
                    ),
                ),
            ),
        )
        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content="Now select the next participant to speak."),
        )

        response = await self.service.get_chat_message_content(
            chat_history,
            settings=PromptExecutionSettings(response_format=StringResult),
        )

        participant_name_with_reason = StringResult.model_validate_json(response.content or "{}")

        print("*********************")
        print(
            f"Next participant: {participant_name_with_reason.result}\nReason: {participant_name_with_reason.reason}."
        )
        print("*********************")

        if participant_name_with_reason.result in participant_descriptions:
            return participant_name_with_reason

        raise RuntimeError(f"Unknown participant selected: {response.content}.")

    @override
    async def filter_results(
        self,
        chat_history: ChatHistory,
    ) -> MessageResult:
        """Provide concrete implementation for filtering the results of the discussion.

        The manager will filter the results of the conversation after the conversation is terminated.
        """
        if not chat_history.messages:
            raise RuntimeError("No messages in the chat history.")

        chat_history.messages.insert(
            0,
            ChatMessageContent(
                role=AuthorRole.SYSTEM,
                content=await self._render_prompt(
                    self.result_filter_prompt,
                    KernelArguments(topic=self.topic),
                ),
            ),
        )
        chat_history.add_message(
            ChatMessageContent(role=AuthorRole.USER, content="Please summarize the discussion."),
        )

        response = await self.service.get_chat_message_content(
            chat_history,
            settings=PromptExecutionSettings(response_format=StringResult),
        )
        string_with_reason = StringResult.model_validate_json(response.content or "{}")

        return MessageResult(
            result=ChatMessageContent(role=AuthorRole.ASSISTANT, content=string_with_reason.result),
            reason=string_with_reason.reason,
        )

async def agent_response_callback(messages) -> None:
    """Callback function to retrieve agent responses."""
    if isinstance(messages, list):
        for message in messages:
            print(f"**{message.name}**\n{message.content}")
    else:
        print(f"**{messages.name}**\n{messages.content}")

async def main():
    """Main function to run the agents."""
    # 1. Create a group chat orchestration with the custom group chat manager
    agents = await get_agents()
    group_chat_orchestration = GroupChatOrchestration(
        members=agents,
        manager=ChatCompletionGroupChatManager(
            topic="How should your government approach taxation?",
            service=AzureChatCompletion(),
            max_rounds=10,
        ),
        agent_response_callback=agent_response_callback,
    )

    # 2. Create a runtime and start it
    runtime = InProcessRuntime()
    runtime.start()

    # 3. Invoke the orchestration with a task and the runtime
    orchestration_result = await group_chat_orchestration.invoke(
        task="Please start the discussion.",
        runtime=runtime,
    )

    # 4. Wait for the results
    value = await orchestration_result.get()
    print(value)

    # 5. Stop the runtime after the invocation is complete
    await runtime.stop_when_idle()

    """
    Sample output:
    *********************
    Should terminate: False
    Reason: The discussion on what a good life means personally has not begun, meaning participants have not yet...
    *********************
    *********************
    Next participant: Farmer
    Reason: The Farmer from Southeast Asia can provide a perspective that highlights the importance of a connection...
    *********************
    **Farmer**
    Thank you for the opportunity to share my perspective. As a farmer from Southeast Asia, my life is intricately...
    *********************
    Should terminate: False
    Reason: The discussion has just started and only one perspective has been shared. There is room for further...
    *********************
    *********************
    Next participant: Developer
    Reason: To provide a contrast between rural and urban perspectives on what constitutes a good life, following the...
    *********************
    **Developer**
    Thank you for the opportunity to join the discussion. As a software developer living in a technology-driven...
    *********************
    Should terminate: False
    Reason: The discussion has just started with perspectives from both a farmer and a developer regarding the...
    *********************
    *********************
    Next participant: Teacher
    Reason: The Teacher, with their extensive experience and historical perspective, can provide valuable insights...
    *********************
    **Teacher**
    As a retired history teacher from Eastern Europe, I find it fascinating to explore how the threads of history,...
    *********************
    Should terminate: True
    Reason: The participants, representing diverse perspectives—a farmer, a developer, and a teacher—have each shared...
    *********************
    Our discussion on what constitutes a good life revolved around key perspectives from a farmer, a developer, and a...
    """

if __name__ == "__main__":
    asyncio.run(main())
