import os
import json
from dotenv import load_dotenv

from azure.identity import ClientSecretCredential
from azure.core.credentials import AzureKeyCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import AzureAISearchTool, ConnectionType
from azure.search.documents import SearchClient
# The response message content objects are part of the inference models
from azure.ai.inference.models import UserMessage, MessageText, MessageTextContent

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
    """Sets up and returns the Azure AI Search tool."""
    if not project_client:
        return None

    try:
        search_conn = project_client.connections.get_default(
            connection_type=ConnectionType.AZURE_AI_SEARCH,
            include_credentials=True
        )

        if not search_conn:
            print("× No default Azure AI Search Connection found in project.")
            return None

        # The AzureAISearchTool helper class correctly sets up the tool for citation generation.
        # No complex field_mapping is needed here; the platform handles it by default.
        ai_search_tool = AzureAISearchTool(
            index_connection_id=search_conn.id,
            index_name=search_index_name,
            # You can still configure top_k or query_type if needed
            top_k=5 
        )
        print(f"✓ Successfully configured AzureAISearchTool for index '{search_index_name}'")
        return ai_search_tool
    except Exception as e:
        print(f"× Error setting up search tool: {str(e)}")
        return None

def create_agent_with_tool(project_client, model_deployment_name, ai_search_tool):
    """Creates an agent with the provided search tool and instructions to generate citations."""
    if not project_client or not ai_search_tool:
        return None
   
    try:
        # --- KEY CHANGE: Updated instructions ---
        # Explicitly instruct the agent to use the tool and cite its sources.
        # This is crucial for triggering the annotation generation.
        instructions = (
            "You are an expert accounting policy assistant. "
            "You must answer questions using only the information available in the search tool. "
            "You MUST cite your sources. For each piece of information you provide, "
            "include a citation reference in your response. The citations will be "
            "automatically generated from the documents you use."
        )

        agent = project_client.agents.create_agent(
            model=model_deployment_name,
            name="IFRS-Search-Agent-Citations", # Using a new name to avoid conflicts
            instructions=instructions,
            tools=ai_search_tool.definitions,
            tool_resources=ai_search_tool.resources,
            headers={"x-ms-enable-preview": "true"}
        )
        print(f"✓ Agent '{agent.name}' created with ID: {agent.id}")
        return agent
    except Exception as e:
        # If an agent with this name already exists, you might get an error.
        # Consider adding logic to find and reuse an existing agent if needed.
        print(f"× Error creating agent: {str(e)}")
        return None

def sdk_object_serializer(o):
    """A custom JSON serializer for SDK objects that are not directly serializable."""
    if hasattr(o, '__dict__'):
        return vars(o)
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def run_agent_query(project_client, agent, question: str):
    """
    Runs a query and prints the raw annotation objects using a custom JSON serializer.
    """
    if not project_client or not agent:
        print("× Cannot run agent query: Client or agent not initialized.")
        return

    try:
        # Steps 1-3 are unchanged
        thread = project_client.agents.create_thread()
        print(f"\n📝 Created thread, ID: {thread.id} for question: '{question}'")

        project_client.agents.create_message(
            thread_id=thread.id,
            role="user",
            content=question
        )

        run = project_client.agents.create_and_process_run(
            thread_id=thread.id,
            agent_id=agent.id
        )
        print(f"🤖 Agent run status: {run.status}")

        if run.last_error:
            print(f"⚠️ Run error: {run.last_error.message}")
            return

        # Step 4: Get the agent's response and inspect annotations
        msg_list = project_client.agents.list_messages(thread_id=thread.id, order_by="created_at desc")
       
        assistant_responded = False
        for m in msg_list.data:
            if m.role == "assistant" and m.content:
                print("\n✅ Assistant Response:")
                
                for content_block in m.content:
                    if hasattr(content_block, "text"):
                        text_value = content_block.text.value
                        annotations = content_block.text.annotations
                        
                        print(text_value)
                        
                        if annotations:
                            print("\n🔍 Raw Annotation Objects (for inspection):")
                            for i, annotation in enumerate(annotations):
                                print(f"--- Annotation [{i+1}] ---")
                                
                                # --- THIS IS THE KEY CHANGE ---
                                # We pass the original annotation object directly to json.dumps
                                # and provide our custom serializer via the 'default' parameter.
                                print(json.dumps(annotation, indent=2, default=sdk_object_serializer))
                                # ------------------------------
                        else:
                            print("\n- No annotations provided for this response.")

                assistant_responded = True
                break
       
        if not assistant_responded:
            print("   - No response from assistant found in the thread.")

    except Exception as e:
        print(f"× An error occurred while running agent query for '{question}': {str(e)}")


def main():
    """Main function to orchestrate the script's operations."""
    load_dotenv('.env')
    project_client = initialize_clients()
    if not project_client:
        return

    model_deployment_name = "gpt-4o-3"
    perform_simple_completion(project_client, model_deployment_name)

    s_index_name = "fin-apd-ifrs-index"
    ai_search_tool = setup_search_tool(project_client, s_index_name)
   
    if not ai_search_tool:
        print("× Agent creation skipped due to search tool setup failure.")
        return

    # Create the agent with instructions for citations
    agent = create_agent_with_tool(project_client, model_deployment_name, ai_search_tool)

    if agent:
        print("\n--- Running Agent Queries ---")
        run_agent_query(project_client, agent, "What is the purpose of IFRS 17?")
        run_agent_query(project_client, agent, "Which insurance contracts does IFRS 17 apply to?")
    else:
        print("× Skipping agent queries as agent creation failed.")

if __name__ == "__main__":
    main()
