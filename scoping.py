import os
from dotenv import load_dotenv
from azure.identity.aio import ClientSecretCredential
from azure.ai.projects.aio import AIProjectClient
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion # For context and Kernel setup

# Load environment variables from .env file (ensure .env is in the execution path or specify path)
load_dotenv() # Looks for .env in current dir or parent dirs

# --- Azure AI Foundry Project Connection Setup ---
# This section initializes the connection to an Azure AI Project.
# The 'azure_credential' and 'project_client' can be used by other parts of the application
# that need to interact with Azure AI Project resources.
#
# For the ReportScopingAgent, its Kernel must be configured with an AzureChatCompletion
# service pointing to a model deployment within this Azure AI Project.
# This typically involves setting AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
# and using 'azure_credential' (or API key) when creating AzureChatCompletion.

AZURE_CLIENT_ID = os.getenv("CLIENT_ID")
AZURE_CLIENT_SECRET = os.getenv("CLIENT_SECRET")
AZURE_TENANT_ID = os.getenv("TENANT_ID")
PROJECT_CONNECTION_STRING = os.getenv("PROJECT_CONNECTION_STRING")

azure_credential = None
project_client = None

if all(var is not None for var in [AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID, PROJECT_CONNECTION_STRING]):
    try:
        azure_credential = ClientSecretCredential(
            client_id=AZURE_CLIENT_ID,
            client_secret=AZURE_CLIENT_SECRET,
            tenant_id=AZURE_TENANT_ID
        )
        # AIProjectClient.from_connection_string is a synchronous method to create the client.
        # Async operations are performed using methods of the client instance.
        project_client = AIProjectClient.from_connection_string(
            credential=azure_credential,
            conn_str=PROJECT_CONNECTION_STRING
        )
        # print(f"Successfully initialized AIProjectClient for project.") # Optional: for debugging
    except Exception as e:
        print(f"Warning: Failed to initialize Azure AI Project client: {e}")
        # azure_credential might be set, but project_client will be None or raise error on use
        project_client = None # Ensure it's None on error
else:
    missing_vars = [
        var_name for var_name, var_value in {
            "CLIENT_ID": AZURE_CLIENT_ID, "CLIENT_SECRET": AZURE_CLIENT_SECRET,
            "TENANT_ID": AZURE_TENANT_ID, "PROJECT_CONNECTION_STRING": PROJECT_CONNECTION_STRING
        }.items() if var_value is None
    ]
    if missing_vars: # Only print warning if there are actually missing variables
        print(f"Warning: Azure AI Foundry environment variables not fully set. Missing: {', '.join(missing_vars)}. "
              "AIProjectClient will not be initialized. Ensure .env file is present and loaded, or variables are set.")

# --- End Azure AI Foundry Project Connection Setup ---

# Original imports of report_scoping_agent.py follow:
import semantic_kernel as sk
from semantic_kernel.kernel import Kernel
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.prompt_template.prompt_template_config import PromptTemplateConfig
from semantic_kernel.prompt_template.input_variable import InputVariable
import json

class ReportScopingAgent:
    """
    An agent responsible for refining a report topic and scope using Semantic Kernel functions.

    Purpose:
    This agent receives an initial user query/topic for a report. 
    If the query is overly broad, it formulates and asks one clarifying question 
    to narrow the scope. It then processes the user's response to the clarifying 
    question (if asked) and produces a "refined topic and scope statement".

    Interaction Flow:
    1. Receives an initial topic.
    2. If the topic is broad and no previous clarification response is given, it generates a clarifying question.
    3. If the topic is specific or a clarification response is provided, it generates a refined topic and scope.

    Inputs to `scope_topic` method:
    - `initial_topic` (str): The user's raw report topic.
    - `clarification_response` (str, optional): User's response to a previous clarifying question.

    Outputs from `scope_topic` method:
    - If clarification is needed: A string containing the clarifying question.
    - If topic is refined: A dictionary with keys "refined_topic" and "scope_statement".
      Example: {"refined_topic": "AI in Financial Reporting", 
                "scope_statement": "Focus on AI applications in automating financial statement preparation."}
    """

    _SYSTEM_MESSAGE = """You are an AI assistant helping to refine a report topic for financial or accounting analysis. Your goal is to produce a "refined topic and scope statement" or ask a clarifying question if needed."""

    _SCOPING_PROMPT_TEMPLATE = """User's initial topic: {{$topic}}
{{if $clarification_response}}
User's response to clarification: {{$clarification_response}}
{{/if}}

Instructions:
1. Analyze the provided topic (and clarification response, if any).
2. If the topic is too broad (e.g., "Report on AI", "Sustainability", "Global Economy") AND no clarification_response is provided, you MUST formulate one specific clarifying question to help narrow the scope. The question should offer 2-3 concrete examples relevant to finance, accounting, or business strategy. 
   Example for "AI": "Could you specify which aspect of AI you're interested in, for example, AI in financial reporting, AI ethics in accounting, or recent AI advancements impacting audits?"
3. If the topic is specific enough, or if a clarification_response has been provided that sufficiently narrows the topic, proceed to step 4.
4. Produce a "refined topic and scope statement". This should be a concise statement that clearly defines the focus of the report.

Output Format:
- If asking a clarifying question (as per instruction 2): Output ONLY the question as a plain string.
- If providing a refined topic and scope (as per instruction 4): Output a JSON object string with two keys: "refined_topic" and "scope_statement".
  Example:
  {
    "refined_topic": "AI in Financial Reporting",
    "scope_statement": "This report will focus on the application of artificial intelligence technologies in automating financial reporting processes, including benefits, challenges, and case studies of current implementations."
  }

Consider the following examples:

Example 1 (Broad Topic - Needs Clarification):
User's initial topic: "AI"
Expected output (Clarifying Question):
Could you specify which aspect of AI you're interested in, for example, AI in financial reporting, AI ethics in accounting, or recent AI advancements impacting audits?

Example 2 (Specific Topic - Direct Refinement):
User's initial topic: "The impact of IFRS 16 on retail industry lease accounting"
Expected output (Refined Scope JSON):
{
  "refined_topic": "Impact of IFRS 16 on Retail Industry Lease Accounting",
  "scope_statement": "This report will analyze the impact of IFRS 16 Leases on the financial statements and key performance indicators of companies in the retail industry, focusing on changes in balance sheets, income statements, and financial ratios."
}

Example 3 (Broad Topic with Clarification Response):
User's initial topic: "AI"
User's response to clarification: "Focus on AI ethics in accounting"
Expected output (Refined Scope JSON):
{
  "refined_topic": "AI Ethics in Accounting",
  "scope_statement": "This report will explore the ethical implications of using artificial intelligence in accounting practices, including issues of bias, transparency, accountability, and data privacy."
}
"""

    def __init__(self, kernel: Kernel):
        """
        Initializes the ReportScopingAgent.

        Args:
            kernel: The Semantic Kernel instance to be used for function invocation.
        """
        self.kernel = kernel

        prompt_template_config = PromptTemplateConfig(
            template=self._SCOPING_PROMPT_TEMPLATE,
            name="ScopeReportTopic",
            description="Refines a report topic, asking for clarification if needed, or producing a refined topic/scope.",
            template_format="semantic-kernel",
            input_variables=[
                InputVariable(name="topic", description="The user's raw report topic", is_required=True),
                InputVariable(name="clarification_response", description="User's response to a clarifying question", is_required=False, default_value=""),
            ]
            # Execution settings can be defined here if they are static for this function,
            # or passed during kernel.invoke() if they are dynamic.
        )

        self.scoping_function = self.kernel.add_function(
            function_name="ScopeReportTopicFunction",
            plugin_name="ReportScopingPlugin",
            prompt_template_config=prompt_template_config,
            prompt=self._SYSTEM_MESSAGE # This sets the system message for the chat completion
        )

    async def scope_topic(self, initial_topic: str, clarification_response: str = None) -> str | dict:
        """
        Processes the topic to refine it or ask a clarifying question.

        This method invokes the underlying Semantic Kernel function with the provided
        topic and optional clarification response. It then parses the LLM's output
        to determine if it's a clarifying question (string) or a refined topic/scope (dict).

        Args:
            initial_topic: The user's raw report topic (e.g., "AI", "Impact of new regulations on banking").
            clarification_response: User's response to a previous clarifying question (if any).
                                      (e.g., "Focus on AI in financial reporting"). Defaults to None.

        Returns:
            - A string containing a clarifying question if the LLM determines the topic is too broad.
              (e.g., "Could you specify which aspect of AI you're interested in...?")
            - A dictionary with "refined_topic" and "scope_statement" keys if the LLM refines the topic.
              (e.g., {"refined_topic": "AI in Financial Reporting", 
                       "scope_statement": "Analysis of AI applications in automating financial statement preparation..."})
            - The raw string output from the LLM if it's neither a valid JSON for refined scope nor clearly a question,
              allowing for inspection of unexpected outputs.
        """
        arguments = KernelArguments(topic=initial_topic)
        if clarification_response:
            arguments["clarification_response"] = clarification_response
        else:
            arguments["clarification_response"] = "" # Ensure empty string for template logic

        # It's assumed the kernel has a chat completion service configured as default or one is specified
        # in prompt_template_config.execution_settings or during kernel.invoke if needed.
        result = await self.kernel.invoke(
            self.scoping_function,
            arguments=arguments
        )
        
        result_str = str(result).strip()

        try:
            # Attempt to parse as JSON (expected for refined scope)
            output_data = json.loads(result_str)
            if isinstance(output_data, dict) and "refined_topic" in output_data and "scope_statement" in output_data:
                return output_data
            # If JSON is valid but not the expected structure, it might be an error or unexpected LLM output.
            # Fall through to return as string.
        except json.JSONDecodeError:
            # If not JSON, it's expected to be a clarifying question (plain string)
            pass 
        
        # If it's not valid, structured JSON, return the string as is (likely a question or unhandled LLM output)
        return result_str


# --- Test Cases for ReportScopingAgent ---
if __name__ == "__main__":
    import asyncio

    async def run_agent_tests():
        print("--- Running ReportScopingAgent Tests ---")

        # Check if Azure credentials and project client were initialized from the top of the file
        # These are assumed to be globally available if the setup code ran successfully.
        if 'azure_credential' not in globals() or azure_credential is None:
            print("Azure credential not initialized. Ensure .env file is correct and Azure setup succeeded.")
            print("Skipping agent tests that require Azure connection.")
            return

        AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")

        if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_CHAT_DEPLOYMENT_NAME:
            print("AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_CHAT_DEPLOYMENT_NAME not set in environment variables.")
            print("Skipping agent tests that require these settings for AzureChatCompletion.")
            return

        try:
            # 1. Initialize Kernel
            kernel = Kernel()

            # 2. Add AzureChatCompletion service using the credential from Azure AI Foundry setup
            # The AzureChatCompletion connector can use the azure_credential directly.
            kernel.add_service(
                AzureChatCompletion(
                    deployment_name=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
                    endpoint=AZURE_OPENAI_ENDPOINT,
                    credential=azure_credential, # Using the credential obtained earlier
                ),
            )
            print("Kernel initialized with AzureChatCompletion service.")
        except Exception as e:
            print(f"Error initializing Kernel or AzureChatCompletion service: {e}")
            return

        # 3. Instantiate the agent
        try:
            scoping_agent = ReportScopingAgent(kernel)
            print("ReportScopingAgent instantiated.")
        except Exception as e:
            print(f"Error instantiating ReportScopingAgent: {e}")
            return

        # --- Test Case 1: Broad Topic --- 
        print("\n--- Test Case 1: Broad Topic ---")
        topic1 = "ESG investing"
        print(f"Input Topic: {topic1}")
        try:
            result1 = await scoping_agent.scope_topic(initial_topic=topic1)
            print(f"Agent Output:\n{result1}")
        except Exception as e:
            print(f"Error during Test Case 1: {e}")

        # --- Test Case 2: Specific Topic --- 
        print("\n--- Test Case 2: Specific Topic ---")
        topic2 = "Analyzing the financial impact of climate-related risks on real estate portfolios in coastal Florida."
        print(f"Input Topic: {topic2}")
        try:
            result2 = await scoping_agent.scope_topic(initial_topic=topic2)
            print(f"Agent Output:\n{result2}")
        except Exception as e:
            print(f"Error during Test Case 2: {e}")

        # --- Test Case 3: Broad Topic with Clarification --- 
        print("\n--- Test Case 3: Broad Topic with Clarification ---")
        topic3_initial = "Cryptocurrency regulation"
        clarification3 = "I'm interested in the evolving SEC guidelines for cryptocurrency exchanges in the US."
        print(f"Input Topic: {topic3_initial}")
        print(f"Clarification Response: {clarification3}")
        try:
            result3 = await scoping_agent.scope_topic(initial_topic=topic3_initial, clarification_response=clarification3)
            print(f"Agent Output:\n{result3}")
        except Exception as e:
            print(f"Error during Test Case 3: {e}")
        
        print("\n--- Agent Tests Completed ---")

    # Run the async test runner
    asyncio.run(run_agent_tests())
