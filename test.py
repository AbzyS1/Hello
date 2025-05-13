# Import required libraries
import os
import json
from dotenv import load_dotenv

from azure.identity import ClientSecretCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool, ConnectionType
from azure.search.documents import SearchClient
from azure.ai.inference.models import UserMessage

def initialize_clients():
    """Initializes and returns the AIProjectClient."""
    try:
        credential = ClientSecretCredential(
            client_id=os.getenv("CLIENT_ID"),
            client_secret=os.getenv("CLIENT_SECRET"),
            tenant_id=os.getenv("TENANT_ID")
        )
        
        project_client = AIProjectClient.from_connection_string(
            conn_str=os.getenv("PROJECT_CONNECTION_STRING"),
            credential=credential
            # api_version="2024-12-01-preview" # Optional: Uncomment if a specific API version is needed
        )
        print("✓ Successfully initialized AIProjectClient")
        return project_client
    except Exception as e:
        print(f"× Error initializing AIProjectClient: {str(e)}")
        return None

def perform_simple_completion(project_client, model_deployment_name="gpt-4o-3"):
    """Performs a simple chat completion."""
    if not project_client:
        return

    try:
        chat_client = project_client.inference.get_chat_completions_client()
        response = chat_client.complete(
            model=model_deployment_name,
            messages=[UserMessage(content="How to be healthy in one sentence?")]
        )
        print("\nSimple Chat Completion Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred during simple chat completion: {str(e)}")

def setup_search_tool(project_client, search_index_name="fin-apd-ifrs-index"):
    """Sets up and returns the Azure AI Search tool and SearchClient."""
    if not project_client:
        return None, None

    try:
        search_conn = project_client.connections.get_default(
            connection_type=ConnectionType.AZURE_AI_SEARCH,
            include_credentials=True
        )

        if not search_conn:
            print("× No default Azure AI Search Connection found in project.")
            return None, None

        s_endpoint = search_conn.endpoint_url
        s_credential = AzureKeyCredential(search_conn.key)

        search_client = SearchClient(
            endpoint=s_endpoint,
            index_name=search_index_name,
            credential=s_credential
        )
        print("✓ Successfully initialized SearchClient")

        # Optional: Perform a test search
        # results = search_client.search(search_text="cash flow", filter=None, top=1)
        # print(f"✓ Test search successful. Found {results.get_count()} documents for 'cash flow'.")

        ai_search_tool = AzureAISearchTool(
            index_connection_id=search_conn.id,
            index_name=search_index_name
        )
        print("✓ Successfully configured AzureAISearchTool")
        return ai_search_tool, search_client
    except Exception as e:
        print(f"× Error setting up search tool: {str(e)}")
        return None, None

def create_agent_with_tool(project_client, model_deployment_name, ai_search_tool):
    """Creates an agent with the provided search tool."""
    if not project_client or not ai_search_tool:
        return None
    
    try:
        agent = project_client.agents.create_agent(
            model=model_deployment_name,
            name="IFRS-Search-Agent",
            instructions="You are an accounting policy expert. You provide the most helpful information only.",
            tools=ai_search_tool.definitions,
            tool_resources=ai_search_tool.resources,
            headers={"x-ms-enable-preview": "true"} # Required if the feature is in preview
        )
        print(f"✓ Agent '{agent.name}' created with ID: {agent.id}")
        return agent
    except Exception as e:
        print(f"× Error creating agent: {str(e)}")
        return None

def run_agent_query(project_client, agent, question: str):
    """Runs a query against the specified agent."""
    if not project_client or not agent:
        print("× Cannot run agent query: Client or agent not initialized.")
        return

    try:
        # Step 1: Create a new conversation thread
        thread = project_client.agents.create_thread()
        print(f"\n📝 Created thread, ID: {thread.id} for question: '{question}'")

        # Step 2: Add the user's question as a message in the thread
        message = project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=question
        )
        print(f"💬 Created user message, ID: {message.id}")

        # Step 3: Create and start an agent run
        run = project_client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent.id
        )
        print(f"🤖 Agent run status: {run.status}")

        if run.last_error:
            print(f"⚠️ Run error: {run.last_error.message}")
            return # Exit if there was an error during the run processing itself

        # Step 4: Get the agent's response
        msg_list = project_client.agents.list_messages(thread_id=thread.id, order_by="created_at desc")
        
        assistant_responded = False
        for m in msg_list.data: # Iterate from newest to oldest
            if m.role == "assistant" and m.content:
                print("\nAssistant says:")
                for c in m.content:
                    if hasattr(c, "text") and c.text:
                        print(c.text.value)
                assistant_responded = True
                break 
        
        if not assistant_responded:
            print("   No response from assistant found in the thread.")

    except Exception as e:
        print(f"× An error occurred while running agent query for '{question}': {str(e)}")


def main():
    """Main function to orchestrate the script's operations."""
    # Load environment variables from .env file
    load_dotenv('.env')

    # Initialize AIProjectClient
    project_client = initialize_clients()
    if not project_client:
        return # Exit if client initialization fails

    # Model deployment name
    model_deployment_name = "gpt-4o-3" # Ensure this model is available in your project

    # Perform a simple completion (optional, as in the original script)
    perform_simple_completion(project_client, model_deployment_name)

    # Setup Azure AI Search tool
    s_index_name = "fin-apd-ifrs-index" # Your specific search index name
    ai_search_tool, _ = setup_search_tool(project_client, s_index_name) # We don't need search_client in main flow after setup
    
    if not ai_search_tool:
        print("× Agent creation skipped due to search tool setup failure.")
        return

    # Create the agent
    agent = create_agent_with_tool(project_client, model_deployment_name, ai_search_tool)

    if agent:
        # Run agent queries
        print("\n--- Running Agent Queries ---")
        run_agent_query(project_client, agent, "Purpose of IFRS17??")
        run_agent_query(project_client, agent, "List of insurance contracts that IFRS17 applies??")
    else:
        print("× Skipping agent queries as agent creation failed.")

if __name__ == "__main__":
    main()
