# report_scoping_agent.py
import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from typing import Optional, Union, Dict, Any

from azure.identity.aio import ClientSecretCredential
from azure.ai.projects.aio import AIProjectClient
from azure.ai.projects.models import Agent as AzureAIAgentModel # For type hinting

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentSettings
from semantic_kernel.contents.chat_message_content import ChatMessageContent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

class ReportScopingAgent:
    """
    An agent responsible for refining a report topic and scope, integrated with Azure AI Foundry.

    This agent uses AzureAIAgent to interact with a model provisioned in Azure AI Foundry.
    It guides the user to refine a broad report topic by asking clarifying questions or,
    if the topic is specific enough, outputs a structured JSON object containing the
    refined topic and scope statement.
    """

    AGENT_NAME_IN_FOUNDRY = "ReportScopingAgentFoundry"
    # General instructions for the agent provisioned in Azure AI Foundry
    AZURE_AI_AGENT_INSTRUCTIONS = """
    You are an AI assistant helping a user scope a research report.
    Your goal is to either:
    1. If the user's input (topic) is specific enough: Output a JSON object with "refined_topic" and "scope_statement".
    2. If the user's input (topic) is too broad: Ask ONE clarifying question to help narrow the scope.

    When responding to a user's topic or their clarification:
    - If the topic is too broad, ask a clarifying question. For example, if the topic is "ESG", you could ask:
      "Are you interested in the environmental, social, or governance aspects of ESG, or a specific industry's ESG performance (e.g., ESG in the Energy sector, ESG in Tech)?"
    - If the topic is specific enough, provide the JSON.

    Provide ONLY the JSON object or ONLY the clarifying question as a string.
    Do not add any preamble or explanation before or after the JSON or question.
    """

    def __init__(self, azure_ai_agent: AzureAIAgent, agent_definition: AzureAIAgentModel):
        """
        Initializes the ReportScopingAgent.

        Args:
            azure_ai_agent: An initialized AzureAIAgent instance.
            agent_definition: The definition of the agent as created in Azure AI Foundry.
        """
        self.azure_agent = azure_ai_agent
        self.agent_definition = agent_definition
        self.thread_id: Optional[str] = None # To maintain conversation context if needed

    @classmethod
    async def create(cls, client: AIProjectClient, model_deployment_name: str) -> 'ReportScopingAgent':
        """
        Asynchronously creates and provisions the agent in Azure AI Foundry,
        then returns an instance of ReportScopingAgent.

        Args:
            client: The AIProjectClient for interacting with Azure AI services.
            model_deployment_name: The name of the model deployment to use for this agent.

        Returns:
            An initialized instance of ReportScopingAgent.
        """
        try:
            # Check if agent already exists by name to avoid re-creating, or decide on update logic
            # For simplicity, this example always tries to create. 
            # A more robust version might list_agents and update if found, or use a unique name.
            logger.info(f"Attempting to create/update agent '{cls.AGENT_NAME_IN_FOUNDRY}' in Azure AI Foundry...")
            agent_definition = await client.agents.create_agent(
                model=model_deployment_name,
                name=cls.AGENT_NAME_IN_FOUNDRY,
                instructions=cls.AZURE_AI_AGENT_INSTRUCTIONS,
                description="Agent to scope research report topics. Asks clarifying questions or outputs refined scope."
            )
            logger.info(f"Agent '{agent_definition.name}' (ID: {agent_definition.id}) created/updated in Azure AI Foundry.")
        except Exception as e:
            logger.error(f"Failed to create/update agent in Azure AI Foundry: {e}")
            # Attempt to get the agent if creation failed due to it already existing (simple recovery)
            try:
                agents = await client.agents.list_agents(name=cls.AGENT_NAME_IN_FOUNDRY)
                existing_agent = next((a for a in agents if a.name == cls.AGENT_NAME_IN_FOUNDRY), None)
                if existing_agent:
                    agent_definition = existing_agent
                    logger.info(f"Using existing agent '{agent_definition.name}' (ID: {agent_definition.id}).")
                else:
                    raise e # Re-raise original exception if not found
            except Exception as fetch_e:
                logger.error(f"Failed to fetch existing agent by name after creation attempt failed: {fetch_e}")
                raise e # Re-raise original creation exception

        azure_ai_agent_instance = AzureAIAgent(client=client, definition=agent_definition)
        return cls(azure_ai_agent=azure_ai_agent_instance, agent_definition=agent_definition)

    async def scope_topic(self, initial_topic: str, clarification_response: Optional[str] = None) -> str:
        """
        Processes the topic to refine it or ask a clarifying question using AzureAIAgent.

        Args:
            initial_topic: The initial topic for the report.
            clarification_response: An optional response from the user to a previous clarifying question.

        Returns:
            A string which is either a JSON object containing the refined topic and scope,
            or a string containing a clarifying question.
        """
        # Construct the message to send to the AzureAIAgent
        # This message will be processed based on the agent's instructions.
        message_parts = [f"Initial Topic: {initial_topic}"]
        if clarification_response:
            message_parts.append(f"User's Response to Clarification: {clarification_response}")
            message_parts.append("Based on this, please provide the JSON object with \"refined_topic\" and \"scope_statement\".")
        else:
            message_parts.append("If this topic is too broad, ask a clarifying question. If it's specific enough, provide the JSON.")
        
        user_message = "\n".join(message_parts)

        try:
            # For simplicity, creating a new thread for each interaction to avoid complex state management.
            # If conversational history is important across scope_topic calls, manage self.thread_id.
            current_thread = await self.azure_agent.create_thread()
            self.thread_id = current_thread.id

            logger.debug(f"Sending message to AzureAIAgent (Thread ID: {self.thread_id}):\n{user_message}")
            
            response_item = await self.azure_agent.get_response(
                messages=[ChatMessageContent(role="user", content=user_message)],
                thread=current_thread
            )
            result_str = str(response_item.message.content).strip()
            logger.debug(f"Received response from AzureAIAgent:\n{result_str}")

            # Clean up the thread after getting the response for this single turn
            await current_thread.delete()
            self.thread_id = None

        except Exception as e:
            logger.error(f"Error invoking AzureAIAgent: {e}")
            return f"Error: Could not process the topic due to an Azure AI Agent error: {e}"

        # Heuristic: If clarification_response was given, we expect JSON.
        # Also, if the result_str starts with '{', assume it's intended to be JSON.
        expect_json = bool(clarification_response) or result_str.startswith("{")

        if expect_json:
            try:
                parsed_json = json.loads(result_str)
                if isinstance(parsed_json, dict) and \
                   "refined_topic" in parsed_json and \
                   "scope_statement" in parsed_json:
                    return json.dumps(parsed_json, indent=2) # Return formatted JSON
                else:
                    logger.warning(f"Output was expected to be JSON with specific keys, but structure is different: {result_str}")
                    return result_str 
            except json.JSONDecodeError:
                logger.warning(f"Output was expected to be JSON but parsing failed: {result_str}")
                return result_str
        else:
            # Not expecting JSON, so it should be a clarifying question
            return result_str

    async def delete_foundry_agent(self, client: AIProjectClient):
        """
        Deletes the agent from Azure AI Foundry.
        """
        if self.agent_definition and self.agent_definition.id:
            try:
                logger.info(f"Deleting agent '{self.agent_definition.name}' (ID: {self.agent_definition.id}) from Azure AI Foundry...")
                await client.agents.delete_agent(agent_id=self.agent_definition.id)
                logger.info(f"Agent '{self.agent_definition.name}' deleted successfully.")
            except Exception as e:
                logger.error(f"Failed to delete agent '{self.agent_definition.name}' from Azure AI Foundry: {e}")
        else:
            logger.warning("Agent definition or ID not available, cannot delete from Foundry.")


# --- Test Cases for ReportScopingAgent ---
async def run_agent_tests():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    logger.info("--- Running ReportScopingAgent Tests (Azure AI Foundry Integration) ---")

    # Load necessary environment variables
    AZURE_CLIENT_ID = os.getenv("CLIENT_ID")
    AZURE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
    AZURE_TENANT_ID = os.getenv("TENANT_ID")
    PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")
    # This will be used as the model_deployment_name for the AzureAIAgent
    AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") 
    # or os.getenv("AZURE_AI_AGENT_MODEL_DEPLOYMENT_NAME") if you prefer that var name

    required_vars = {
        "CLIENT_ID": AZURE_CLIENT_ID,
        "CLIENT_SECRET": AZURE_CLIENT_SECRET,
        "TENANT_ID": AZURE_TENANT_ID,
        "PROJECT_CONNECTION_STRING": PROJECT_CONNECTION_STRING,
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME": AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
    }

    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}. Ensure .env file is present and loaded.")
        logger.error("Skipping agent tests that require Azure connection.")
        return

    azure_credential = None
    scoping_agent: Optional[ReportScopingAgent] = None
    client: Optional[AIProjectClient] = None

    try:
        azure_credential = ClientSecretCredential(
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET,
            tenant_id=AZURE_TENANT_ID
        )
        logger.info("Azure credential initialized.")

        # Create AIProjectClient using AzureAIAgent.create_client for proper async management
        # Note: AzureAIAgent.create_client expects conn_str if not using default env var for it.
        # We pass PROJECT_CONNECTION_STRING directly.
        client = AzureAIAgent.create_client(credential=azure_credential, conn_str=PROJECT_CONNECTION_STRING)
        await client.__aenter__() # Manually enter async context if not using 'async with'
        logger.info("AIProjectClient initialized.")

        # Instantiate the agent using the async factory method
        scoping_agent = await ReportScopingAgent.create(
            client=client, 
            model_deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME
        )
        logger.info(f"ReportScopingAgent '{scoping_agent.AGENT_NAME_IN_FOUNDRY}' instantiated and provisioned.")

        # --- Test Case 1: Broad Topic --- 
        logger.info("\n--- Test Case 1: Broad Topic ---")
        topic1 = "ESG investing"
        logger.info(f"Input Topic: {topic1}")
        result1 = await scoping_agent.scope_topic(initial_topic=topic1)
        logger.info(f"Agent Output:\n{result1}")

        # --- Test Case 2: Specific Topic --- 
        logger.info("\n--- Test Case 2: Specific Topic ---")
        topic2 = "Analyzing the financial impact of climate-related risks on real estate portfolios in coastal Florida."
        logger.info(f"Input Topic: {topic2}")
        result2 = await scoping_agent.scope_topic(initial_topic=topic2)
        logger.info(f"Agent Output:\n{result2}")

        # --- Test Case 3: Broad Topic with Clarification --- 
        logger.info("\n--- Test Case 3: Broad Topic with Clarification ---")
        topic3_initial = "Cryptocurrency regulation"
        clarification3 = "I'm interested in the evolving SEC guidelines for cryptocurrency exchanges in the US."
        logger.info(f"Input Topic: {topic3_initial}")
        logger.info(f"Clarification Response: {clarification3}")
        result3 = await scoping_agent.scope_topic(initial_topic=topic3_initial, clarification_response=clarification3)
        logger.info(f"Agent Output:\n{result3}")
        
    except Exception as e:
        logger.error(f"An error occurred during agent tests: {e}", exc_info=True)
    finally:
        if scoping_agent and client:
            logger.info("Cleaning up: Deleting agent from Azure AI Foundry...")
            #await scoping_agent.delete_foundry_agent(client)
            pass
        
        if client:
            await client.__aexit__(None, None, None) # Manually exit async context
            logger.info("AIProjectClient closed.")
        if azure_credential:
            await azure_credential.close()
            logger.info("Azure credential closed.")
        
        logger.info("\n--- Agent Tests Completed ---")

if __name__ == "__main__":
    asyncio.run(run_agent_tests())
