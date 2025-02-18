**Lab 0: Environment Setup and Deployment**

### Overview

In this lab, you'll set up the required Azure resources and configure your local development environment for building a multi-agent banking system.

### Learning Objectives

- Deploy required Azure services using `azd`
- Configure local development environment
- Validate the setup

### Prerequisites

- Azure subscription with owner permissions
- Azure CLI installed
- Azure Developer CLI (`azd`) installed
- Git installed
- Visual Studio Code
- Python 3.9 or higher

### Exercise Steps

1. **Clone the Workshop Repository**

   ```bash
   git clone https://github.com/aayush3011/banking-multi-agent-workshop.git
   cd banking-multi-agent-workshop
   git checkout start
   ```

2. **Install Required Tools**

   ```bash
   # Install Azure CLI (if not already installed)
   curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

   # Install Azure Developer CLI
   curl -fsSL https://aka.ms/install-azd.sh | bash
   ```

3. **Azure Resource Deployment**

   ```bash
   # Login to Azure
   az login

   # Set your subscription
   az account set --subscription <your-subscription-id>

   # Initialize and deploy using azd
   azd init
   azd up
   ```

   This will deploy:

   - Azure OpenAI service
   - Azure Cosmos DB
   - Azure App Service (for later labs)

4. **Configure Local Environment**

   ```bash
   # Create and activate virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install dependencies
   pip install -r requirements.txt
   ```

5. **Validate Setup**
   ```python
   # Run the validation script
   python validate_setup.py
   ```
   This will check:
   - Azure OpenAI connectivity
   - Cosmos DB connectivity
   - Required packages installation

### Validation

Your setup is successful if:

- [ ] All Azure resources are deployed successfully
- [ ] Validation script runs without errors
- [ ] You can access Azure OpenAI service
- [ ] You can connect to Cosmos DB

### Common Issues and Troubleshooting

1. Azure OpenAI deployment issues:

   - Ensure your subscription has access to Azure OpenAI
   - Check regional availability

2. Cosmos DB connectivity:

   - Verify connection string in local.settings.json
   - Check firewall settings

3. Python environment issues:
   - Ensure correct Python version
   - Verify all dependencies are installed

### Next Steps

Once you've completed this setup, you're ready to move on to Lab 1, where you'll create your first AI agent.

### Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/cognitive-services/openai/)
- [Azure Cosmos DB Documentation](https://learn.microsoft.com/azure/cosmos-db/)
- [azd Command Reference](https://learn.microsoft.com/azure/developer/azure-developer-cli/reference)

# Lab 1: Creating Your First Banking Agent

## Overview

In this lab, you'll implement the foundation of a multi-agent banking system using LangGraph, starting with a customer support agent. You'll set up Azure services, create basic banking tools, and implement state management using Cosmos DB.

## Learning Objectives

- Set up Azure OpenAI and Cosmos DB integrations
- Create a customer support agent using LangGraph
- Implement basic banking tools
- Create state management with Cosmos DB checkpointing
- Test the agent through a CLI interface

## Prerequisites

- Completed Lab 0 (Environment Setup)
- Python 3.9+
- Azure OpenAI access
- Azure Cosmos DB instance

## Exercise Steps

### Step 1: Environment Setup

Set up the required environment variables:

```bash
# Azure OpenAI Configuration
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_MODEL="gpt-4"

# Cosmos DB Configuration
export COSMOS_ENDPOINT="your-cosmos-endpoint"
export COSMOS_KEY="your-cosmos-key"
export COSMOS_DATABASE="banking"
export COSMOS_CONTAINER="chat_history"
```

### Step 2: Configure Azure Services

1. **Azure OpenAI Configuration**
   Create `src/app/azure_open_ai.py`:

   ```python
   from openai import AsyncAzureOpenAI
   import os

   model = AsyncAzureOpenAI(
       api_key=os.getenv("AZURE_OPENAI_API_KEY"),
       api_version="2024-02-15-preview",
       azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
   )
   ```

2. **Cosmos DB Configuration**
   Create `src/app/azure_cosmos_db.py`:

   ```python
   from azure.cosmos.aio import CosmosClient
   import os

   DATABASE_NAME = os.getenv("COSMOS_DATABASE", "banking")
   CONTAINER_NAME = os.getenv("COSMOS_CONTAINER", "chat_history")

   client = CosmosClient(
       url=os.getenv("COSMOS_ENDPOINT"),
       credential=os.getenv("COSMOS_KEY")
   )

   database = client.get_database_client(DATABASE_NAME)
   container = database.get_container_client(CONTAINER_NAME)
   ```

### Step 3: Implement Banking Tools

Create `src/app/banking.py`:

```python
from typing import Dict, Any

def get_product_advise() -> Dict[str, Any]:
    """Get basic information about banking products."""
    return {
        "accounts": {
            "checking": "Basic checking account with no monthly fees",
            "savings": "High-yield savings account with 3.5% APY",
            "premium": "Premium checking with added benefits"
        },
        "cards": {
            "basic": "No annual fee credit card",
            "rewards": "Cash back rewards card",
            "premium": "Travel rewards card with lounge access"
        }
    }

def get_branch_location() -> Dict[str, Any]:
    """Get information about bank branches."""
    return {
        "locations": [
            {
                "name": "Main Branch",
                "address": "123 Banking St, Financial District",
                "hours": "9 AM - 5 PM",
                "services": ["Full Service", "ATM", "Safe Deposit"]
            },
            {
                "name": "West Side Branch",
                "address": "456 Commerce Ave, West Side",
                "hours": "9 AM - 6 PM",
                "services": ["Full Service", "ATM"]
            }
        ]
    }
```

### Step 4: Create Banking Agents

Create `src/app/banking_agents.py`:

```python
from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, START
from langgraph.prebuilt.agents import create_react_agent
from langgraph_checkpoint_cosmosdb import CosmosDBSaver

from .banking import get_product_advise, get_branch_location
from .azure_open_ai import model
from .azure_cosmos_db import DATABASE_NAME, CONTAINER_NAME

# Define tools for customer support agent
customer_support_agent_tools = [
    get_product_advise,
    get_branch_location,
]

# Create customer support agent
customer_support_agent = create_react_agent(
    model,
    customer_support_agent_tools,
    state_modifier=(
        "You are a customer support agent that can give general advice on banking products and branch locations. "
        "Use the tools available to provide accurate information about our products and branches. "
        "Be professional and courteous in your responses."
    ),
)

class MessagesState(TypedDict):
    messages: List[Dict[str, str]]
    current_agent: str

async def call_customer_support_agent(state: MessagesState) -> MessagesState:
    result = await customer_support_agent.ainvoke(state)
    return result

async def human_node(state: MessagesState) -> MessagesState:
    return state

# Create state graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("customer_support_agent", call_customer_support_agent)
builder.add_node("human", human_node)

# Add edges
builder.add_edge(START, "customer_support_agent")

# Set up checkpointing
checkpointer = CosmosDBSaver(
    database_name=DATABASE_NAME,
    container_name=CONTAINER_NAME
)

# Compile graph
graph = builder.compile(checkpointer=checkpointer)
```

### Step 5: Create Test CLI

Create `test/test_agent.py`:

```python
import asyncio
import sys
import uuid
sys.path.append("../src/app")

from banking_agents import graph

async def test_conversation():
    conversation_id = str(uuid.uuid4())

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        response = await graph.acall({
            "messages": [{"role": "user", "content": user_input}],
            "current_agent": "customer_support_agent",
            "conversation_id": conversation_id
        })

        print(f"Agent: {response['messages'][-1]['content']}")

if __name__ == "__main__":
    asyncio.run(test_conversation())
```

## Testing the Implementation

1. Ensure all environment variables are set correctly
2. Run the test CLI:

   ```bash
   python test/test_agent.py
   ```

3. Try these sample queries:

   ```
   You: What types of accounts do you offer?
   You: Where is your nearest branch?
   You: What are the branch working hours?
   You: Tell me about your credit cards
   ```

4. Type `exit` to end the conversation.

## Validation Checklist

Your implementation is successful if:

- [ ] Agent provides accurate product information using the `get_product_advise` tool
- [ ] Branch location queries are answered correctly using the `get_branch_location` tool
- [ ] Responses are professional and helpful
- [ ] Conversation state is properly saved in Cosmos DB
- [ ] CLI interface works smoothly

## Common Issues and Troubleshooting

1. Tool Integration:

   - Verify tool functions return proper data structures
   - Check tool access in agent responses

2. State Management:

   - Verify Cosmos DB connection string
   - Check checkpoint saving functionality
   - Ensure proper database and container names

3. Agent Responses:

   - Ensure proper tool usage in responses
   - Verify response formatting

4. Environment Setup:
   - Check all environment variables are set
   - Verify Azure OpenAI model deployment
   - Confirm Cosmos DB container exists

## Next Steps

In Lab 2, we'll:

- Add agent transfer capabilities
- Implement the sales and transaction agents
- Create more sophisticated banking tools

# Lab 2: Implementing Multi-Agent Banking System

## Environment Setup

First, set up your environment variables. Create a `.env` file in your project root:

```bash
# Azure OpenAI Configuration
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="your-endpoint"
export AZURE_OPENAI_MODEL="gpt-4"

# Cosmos DB Configuration
export COSMOS_ENDPOINT="your-cosmos-endpoint"
export COSMOS_KEY="your-cosmos-key"
export COSMOS_DATABASE="banking"
export COSMOS_CONTAINER="chat_history"
```

Load these variables:

```bash
source .env
```

## Project Structure

```
src/
├── app/
│   ├── services/
│   │   ├── azure_open_ai.py
│   │   └── azure_cosmos_db.py
│   ├── tools/
│   │   ├── agent_transfers.py
│   │   ├── banking.py
│   │   └── product.py
│   └── banking_agents.py
└── test/
    └── test_agent.py
```

## Implementation

### 1. Azure Services Configuration

Create `src/app/services/azure_open_ai.py`:

```python
from openai import AsyncAzureOpenAI
import os

model = AsyncAzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-02-15-preview",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)
```

Create `src/app/services/azure_cosmos_db.py`:

```python
from azure.cosmos.aio import CosmosClient
import os

DATABASE_NAME = os.getenv("COSMOS_DATABASE", "banking")
CONTAINER_NAME = os.getenv("COSMOS_CONTAINER", "chat_history")

client = CosmosClient(
    url=os.getenv("COSMOS_ENDPOINT"),
    credential=os.getenv("COSMOS_KEY")
)

database = client.get_database_client(DATABASE_NAME)
container = database.get_container_client(CONTAINER_NAME)
```

### 2. Banking Tools

Create `src/app/tools/banking.py`:

```python
from typing import Dict, Any
from datetime import datetime

def bank_balance(account_id: str) -> Dict[str, Any]:
    """Get account balance information."""
    # Hardcoded data for demo
    accounts = {
        "1234": {"balance": 2500.00, "type": "checking"},
        "5678": {"balance": 10000.00, "type": "savings"}
    }
    return accounts.get(account_id, {"error": "Account not found"})

def bank_transfer(from_account: str, to_account: str, amount: float) -> Dict[str, Any]:
    """Process a bank transfer."""
    if amount <= 0:
        return {"error": "Invalid amount"}
    if from_account not in ["1234", "5678"]:
        return {"error": "Source account not found"}
    if to_account not in ["1234", "5678"]:
        return {"error": "Destination account not found"}

    return {
        "status": "success",
        "transaction_id": f"TX_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "amount": amount,
        "from_account": from_account,
        "to_account": to_account
    }

def calculate_monthly_payment(loan_amount: float, years: int) -> Dict[str, Any]:
    """Calculate monthly loan payment."""
    rate = 0.05  # 5% annual interest rate
    monthly_rate = rate / 12
    num_payments = years * 12

    monthly_payment = (loan_amount * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)

    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(monthly_payment * num_payments, 2),
        "interest_rate": f"{rate*100}%"
    }

def create_account(holder_name: str, initial_balance: float) -> Dict[str, Any]:
    """Create a new bank account."""
    account_id = "ACC_" + datetime.now().strftime('%Y%m%d%H%M%S')
    return {
        "status": "success",
        "account_id": account_id,
        "holder_name": holder_name,
        "initial_balance": initial_balance
    }
```

### 3. Product Tools

Create `src/app/tools/product.py`:

```python
from typing import Dict, Any

def get_product_advise() -> Dict[str, Any]:
    """Get basic information about banking products."""
    return {
        "accounts": {
            "checking": "Basic checking account with no monthly fees",
            "savings": "High-yield savings account with 3.5% APY",
            "premium": "Premium checking with added benefits"
        },
        "cards": {
            "basic": "No annual fee credit card",
            "rewards": "Cash back rewards card",
            "premium": "Travel rewards card with lounge access"
        }
    }

def get_branch_location() -> Dict[str, Any]:
    """Get information about bank branches."""
    return {
        "locations": [
            {
                "name": "Main Branch",
                "address": "123 Banking St, Financial District",
                "hours": "9 AM - 5 PM",
                "services": ["Full Service", "ATM", "Safe Deposit"]
            },
            {
                "name": "West Side Branch",
                "address": "456 Commerce Ave, West Side",
                "hours": "9 AM - 6 PM",
                "services": ["Full Service", "ATM"]
            }
        ]
    }
```

### 4. Agent Transfer Tool

Create `src/app/tools/agent_transfers.py`:

```python
from langchain_core.tools import tool
from typing import Annotated
from langchain_core.tools.base import InjectedToolCallId
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.types import Command

def create_agent_transfer(*, agent_name: str):
    """Create a tool that can return handoff via a Command"""
    tool_name = f"transfer_to_{agent_name}"

    @tool(tool_name)
    def transfer_to_agent(
            state: Annotated[dict, InjectedState],
            tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Ask another agent for help."""
        tool_message = {
            "role": "tool",
            "content": f"Successfully transferred to {agent_name}",
            "name": tool_name,
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto=agent_name,
            graph=Command.PARENT,
            update={"messages": state["messages"] + [tool_message]},
        )

    return transfer_to_agent
```

### 5. Banking Agents

Create `src/app/banking_agents.py`:

```python
from typing import Literal
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from langgraph_checkpoint_cosmosdb import CosmosDBSaver

from .services.azure_open_ai import model
from .services.azure_cosmos_db import DATABASE_NAME, CONTAINER_NAME
from .tools.product import get_product_advise, get_branch_location
from .tools.banking import bank_balance, bank_transfer, calculate_monthly_payment, create_account
from .tools.agent_transfers import create_agent_transfer

# Coordinator Agent
coordinator_agent_tools = [
    create_agent_transfer(agent_name="customer_support_agent"),
    create_agent_transfer(agent_name="sales_agent"),
    create_agent_transfer(agent_name="transactions_agent"),
]

coordinator_agent = create_react_agent(
    model,
    coordinator_agent_tools,
    state_modifier=(
        "You are a Chat Initiator and Request Router in a bank. "
        "Your primary responsibilities include welcoming users, and routing requests to the appropriate agent. "
        "If the user needs general help, transfer to 'customer_support_agent' for help. "
        "If the user wants to open a new account or take out a bank loan, transfer to 'sales_agent'. "
        "If the user wants to check their account balance or make a bank transfer, transfer to 'transactions_agent'. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)

# Customer Support Agent
customer_support_agent_tools = [
    get_product_advise,
    get_branch_location,
    create_agent_transfer(agent_name="sales_agent"),
    create_agent_transfer(agent_name="transactions_agent"),
]

customer_support_agent = create_react_agent(
    model,
    customer_support_agent_tools,
    state_modifier=(
        "You are a customer support agent that can give general advice on banking products and branch locations. "
        "If the user wants to open a new account or take out a bank loan, transfer to 'sales_agent'. "
        "If the user wants to check their account balance or make a bank transfer, transfer to 'transactions_agent'. "
        "You MUST include human-readable response before transferring to another agent."
    ),
)

# Sales Agent
sales_agent_tools = [
    calculate_monthly_payment,
    create_account,
    create_agent_transfer(agent_name="customer_support_agent"),
]

sales_agent = create_react_agent(
    model,
    sales_agent_tools,
    state_modifier=(
        "You are a sales agent that can help users with creating a new account, or taking out bank loans. "
        "If the user wants to create a new account, you must ask for the account holder's name and the initial balance. "
        "Call create_account tool with these values. "
        "If user wants to take out a loan, you must ask for the loan amount and the number of years for the loan. "
        "When user provides these, calculate the monthly payment using calculate_monthly_payment tool and provide the result. "
        "Do not return the monthly payment tool call output directly to the user, include it with the rest of your response. "
        "You MUST respond with the repayment amounts before transferring to another agent."
    ),
)

# Transaction Agent
transactions_agent_tools = [
    bank_balance,
    bank_transfer,
    create_agent_transfer(agent_name="customer_support_agent"),
]

transactions_agent = create_react_agent(
    model,
    transactions_agent_tools,
    state_modifier=(
        "You are a banking transactions agent that can handle account balance enquiries and bank transfers. "
        "If the user needs general help, transfer to 'customer_support_agent' for help. "
        "You MUST respond with the transaction details before transferring to another agent."
    ),
)

def call_coordinator_agent(state: MessagesState, config) -> Command[Literal["coordinator_agent", "human"]]:
    response = coordinator_agent.invoke(state)
    return Command(update=response, goto="human")

def call_customer_support_agent(state: MessagesState, config) -> Command[Literal["customer_support_agent", "human"]]:
    response = customer_support_agent.invoke(state)
    return Command(update=response, goto="human")

def call_sales_agent(state: MessagesState, config) -> Command[Literal["sales_agent", "human"]]:
    response = sales_agent.invoke(state)
    return Command(update=response, goto="human")

def call_transactions_agent(state: MessagesState, config) -> Command[Literal["transactions_agent", "human"]]:
    response = transactions_agent.invoke(state)
    return Command(update=response, goto="human")

def human_node(state: MessagesState, config) -> Command[
    Literal["coordinator_agent", "customer_support_agent", "sales_agent", "transactions_agent", "human"]]:
    """A node for collecting user input."""
    user_input = interrupt(value="Ready for user input.")
    langgraph_triggers = config["metadata"]["langgraph_triggers"]
    if len(langgraph_triggers) != 1:
        raise AssertionError("Expected exactly 1 trigger in human node")
    active_agent = langgraph_triggers[0].split(":")[1]
    return Command(update={"messages": [{"role": "human", "content": user_input}]}, goto=active_agent)

# Create state graph
builder = StateGraph(MessagesState)

# Add nodes
builder.add_node("coordinator_agent", call_coordinator_agent)
builder.add_node("customer_support_agent", call_customer_support_agent)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("transactions_agent", call_transactions_agent)
builder.add_node("human", human_node)

# Add edges
builder.add_edge(START, "coordinator_agent")

# Set up checkpointing
checkpointer = CosmosDBSaver(
    database_name=DATABASE_NAME,
    container_name=CONTAINER_NAME
)

# Compile graph
graph = builder.compile(checkpointer=checkpointer)
```

### 6. Test CLI

Create `test/test_agent.py`:

```python
import asyncio
import sys
import uuid
sys.path.append("../src/app")

from banking_agents import graph

async def test_conversation():
    conversation_id = str(uuid.uuid4())

    print("\nBanking System Test CLI")
    print("Type 'exit' to end the conversation")
    print("\nTest scenarios:")
    print("1. General inquiry: 'What banking services do you offer?'")
    print("2. Account creation: 'I want to open a new account'")
    print("3. Balance check: 'What's my account balance for account 1234?'")
    print("4. Transfer money: 'I want to transfer $100 from account 1234 to 5678'")
    print("5. Loan inquiry: 'I want to take out a home loan'")
    print("6. Branch location: 'Where is your nearest branch?'\n")

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        try:
            response = await graph.acall({
                "messages": [{"role": "user", "content": user_input}],
                "conversation_id": conversation_id
            })

            # Extract the last message from the response
            last_message = response["messages"][-1]["content"]
            print(f"Agent: {last_message}")

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_conversation())
```

## Testing

1. Run the test CLI:

```bash
python test/test_agent.py
```

2. Try these test scenarios in sequence:

```
# Test Coordinator Agent
You: Hi, I need some banking help
Expected: Welcome message and routing question

# Test Customer Support
You: What types of accounts do you offer?
You: Where is your nearest branch?

# Test Sales Agent
You: I want to open a new account
You: I want to take out a loan for $200,000 over 30 years

# Test Transaction Agent
You: What's my balance in account 1234?
You: I want to transfer $100 from account 1234 to 5678

# Test Agent Transfers
You: I need help with my account (should route to customer support)
You: I want to open a new account (should route to sales)
You: I want to check my balance (should route to transactions)
```

## Validation Checklist

- [ ] All agents initialize correctly
- [ ] Coordinator routes requests appropriately
- [ ] Tools return expected responses
- [ ] Agent transfers work smoothly
- [ ] State is maintained between interactions
- [ ] Error handling works as expected

## Common Issues and Solutions

1. Environment Variables:

   - Double-check all environment variables are set
   - Verify Azure OpenAI endpoint is accessible
   - Confirm Cosmos DB connection string is valid

2. Agent Routing:

   - Verify coordinator agent responses
   - Check transfer tool implementation
   - Monitor agent state during transfers

3. Tool Execution:

   - Validate tool return formats
   - Check error handling in tools
   - Verify tool access permissions

4. State Management:
   - Monitor Cosmos DB connections
   - Check state preservation
   - Verify conversation flow

## Next Steps

In Lab 3, we'll:

- Implement real banking operations
- Add proper data persistence
- Create robust error handling
- Add transaction validation

# Lab 3: Implementing Core Banking Operations

## Overview

In this lab, you'll implement real banking operations with proper data persistence using Cosmos DB. You'll create robust account management and transaction processing systems with proper validation and error handling.

### Step 1: Update Cosmos DB Configuration

[Previous Cosmos DB code remains the same]

### Step 2: Create Banking Operations

Create `src/app/banking_operations.py`:

```python
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
from decimal import Decimal
from .services.azure_cosmos_db import cosmos_db

class BankingOperations:
    @staticmethod
    def _validate_amount(amount: float) -> bool:
        """Validate if amount is positive and has valid decimal places."""
        try:
            if amount <= 0:
                return False
            # Ensure no more than 2 decimal places
            decimal_amount = Decimal(str(amount))
            return decimal_amount.as_tuple().exponent >= -2
        except:
            return False

    @staticmethod
    async def create_account(
        holder_name: str,
        account_type: str,
        initial_deposit: float
    ) -> Dict[str, Any]:
        """Create a new bank account."""
        try:
            if not BankingOperations._validate_amount(initial_deposit):
                return {"status": "error", "message": "Invalid initial deposit amount"}

            account_id = str(uuid.uuid4())
            account = {
                "id": account_id,
                "account_id": account_id,
                "holder_name": holder_name,
                "account_type": account_type.lower(),
                "balance": initial_deposit,
                "status": "active",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }

            await cosmos_db.accounts.create_item(body=account)
            return {
                "status": "success",
                "account_id": account_id,
                "message": f"Account created successfully for {holder_name}"
            }
        except Exception as e:
            return {"status": "error", "message": f"Error creating account: {str(e)}"}

    @staticmethod
    async def get_account_balance(account_id: str) -> Dict[str, Any]:
        """Get account balance and details."""
        try:
            account = await cosmos_db.accounts.read_item(
                item=account_id,
                partition_key=account_id
            )
            return {
                "status": "success",
                "balance": account["balance"],
                "account_type": account["account_type"],
                "holder_name": account["holder_name"]
            }
        except Exception as e:
            return {"status": "error", "message": "Account not found"}

    @staticmethod
    async def process_transfer(
        from_account: str,
        to_account: str,
        amount: float,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a transfer between accounts."""
        try:
            # Validate amount
            if not BankingOperations._validate_amount(amount):
                return {"status": "error", "message": "Invalid transfer amount"}

            # Get source account
            try:
                source = await cosmos_db.accounts.read_item(
                    item=from_account,
                    partition_key=from_account
                )
            except:
                return {"status": "error", "message": "Source account not found"}

            # Check sufficient funds
            if source["balance"] < amount:
                return {"status": "error", "message": "Insufficient funds"}

            # Get destination account
            try:
                dest = await cosmos_db.accounts.read_item(
                    item=to_account,
                    partition_key=to_account
                )
            except:
                return {"status": "error", "message": "Destination account not found"}

            # Create transaction record
            transaction_id = str(uuid.uuid4())
            transaction = {
                "id": transaction_id,
                "transaction_id": transaction_id,
                "from_account": from_account,
                "to_account": to_account,
                "amount": amount,
                "description": description or "Transfer",
                "status": "completed",
                "timestamp": datetime.utcnow().isoformat()
            }

            # Update account balances
            source["balance"] -= amount
            dest["balance"] += amount
            source["updated_at"] = datetime.utcnow().isoformat()
            dest["updated_at"] = datetime.utcnow().isoformat()

            # Save all changes
            await cosmos_db.transactions.create_item(body=transaction)
            await cosmos_db.accounts.replace_item(
                item=source["id"],
                body=source
            )
            await cosmos_db.accounts.replace_item(
                item=dest["id"],
                body=dest
            )

            return {
                "status": "success",
                "transaction_id": transaction_id,
                "new_balance": source["balance"],
                "message": f"Successfully transferred ${amount} to account {to_account}"
            }
        except Exception as e:
            return {"status": "error", "message": f"Error processing transfer: {str(e)}"}

    @staticmethod
    async def get_transaction_history(
        account_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Get transaction history for an account."""
        try:
            query = """
                SELECT * FROM c
                WHERE c.from_account = @account_id OR c.to_account = @account_id
                ORDER BY c.timestamp DESC
                OFFSET 0 LIMIT @limit
            """
            parameters = [
                {"name": "@account_id", "value": account_id},
                {"name": "@limit", "value": limit}
            ]

            transactions = []
            async for transaction in cosmos_db.transactions.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ):
                transactions.append(transaction)

            return {
                "status": "success",
                "transactions": transactions,
                "count": len(transactions)
            }
        except Exception as e:
            return {"status": "error", "message": f"Error fetching transactions: {str(e)}"}
```

### Step 3: Update Banking Tools

Update `src/app/tools/banking.py`:

```python
from typing import Dict, Any
from ..banking_operations import BankingOperations

async def create_account(holder_name: str, account_type: str, initial_deposit: float) -> Dict[str, Any]:
    """Create a new bank account."""
    return await BankingOperations.create_account(
        holder_name=holder_name,
        account_type=account_type,
        initial_deposit=initial_deposit
    )

async def get_account_balance(account_id: str) -> Dict[str, Any]:
    """Get account balance and details."""
    return await BankingOperations.get_account_balance(account_id)

async def process_transfer(from_account: str, to_account: str, amount: float) -> Dict[str, Any]:
    """Process a transfer between accounts."""
    return await BankingOperations.process_transfer(
        from_account=from_account,
        to_account=to_account,
        amount=amount
    )

async def get_transaction_history(account_id: str, limit: int = 10) -> Dict[str, Any]:
    """Get transaction history for an account."""
    return await BankingOperations.get_transaction_history(
        account_id=account_id,
        limit=limit
    )
```

### Step 4: Create Test Script

Create `test/test_banking_operations.py`:

```python
import asyncio
import sys
sys.path.append("../src/app")

from banking_operations import BankingOperations
from services.azure_cosmos_db import cosmos_db

async def test_banking_operations():
    # Initialize Cosmos DB
    print("Initializing Cosmos DB...")
    await cosmos_db.initialize()

    # Test account creation
    print("\nCreating test accounts...")
    account1 = await BankingOperations.create_account(
        "John Doe",
        "checking",
        1000.0
    )
    print(f"Account 1 creation: {account1}")

    account2 = await BankingOperations.create_account(
        "Jane Doe",
        "savings",
        500.0
    )
    print(f"Account 2 creation: {account2}")

    if account1["status"] == "success" and account2["status"] == "success":
        account1_id = account1["account_id"]
        account2_id = account2["account_id"]

        # Test balance check
        print("\nChecking balances...")
        balance1 = await BankingOperations.get_account_balance(account1_id)
        balance2 = await BankingOperations.get_account_balance(account2_id)
        print(f"Account 1 balance: {balance1}")
        print(f"Account 2 balance: {balance2}")

        # Test transfer
        print("\nProcessing transfer...")
        transfer = await BankingOperations.process_transfer(
            account1_id,
            account2_id,
            200.0,
            "Test transfer"
        )
        print(f"Transfer result: {transfer}")

        # Check updated balances
        print("\nChecking updated balances...")
        balance1 = await BankingOperations.get_account_balance(account1_id)
        balance2 = await BankingOperations.get_account_balance(account2_id)
        print(f"Account 1 new balance: {balance1}")
        print(f"Account 2 new balance: {balance2}")

        # Get transaction history
        print("\nGetting transaction history...")
        history = await BankingOperations.get_transaction_history(account1_id)
        print(f"Transaction history: {history}")

        # Test error cases
        print("\nTesting error cases...")
        # Invalid amount
        invalid_transfer = await BankingOperations.process_transfer(
            account1_id,
            account2_id,
            -100.0
        )
        print(f"Invalid amount transfer: {invalid_transfer}")

        # Insufficient funds
        large_transfer = await BankingOperations.process_transfer(
            account1_id,
            account2_id,
            10000.0
        )
        print(f"Insufficient funds transfer: {large_transfer}")

        # Invalid account
        invalid_account = await BankingOperations.get_account_balance("invalid_id")
        print(f"Invalid account balance check: {invalid_account}")

if __name__ == "__main__":
    asyncio.run(test_banking_operations())
```

## Testing

1. Run the test script:

```bash
python test/test_banking_operations.py
```

2. Expected Output:

```
Initializing Cosmos DB...

Creating test accounts...
Account 1 creation: {'status': 'success', 'account_id': '...', 'message': 'Account created successfully for John Doe'}
Account 2 creation: {'status': 'success', 'account_id': '...', 'message': 'Account created successfully for Jane Doe'}

Checking balances...
Account 1 balance: {'status': 'success', 'balance': 1000.0, 'account_type': 'checking', 'holder_name': 'John Doe'}
Account 2 balance: {'status': 'success', 'balance': 500.0, 'account_type': 'savings', 'holder_name': 'Jane Doe'}

Processing transfer...
Transfer result: {'status': 'success', 'transaction_id': '...', 'new_balance': 800.0, 'message': 'Successfully transferred $200.0 to account ...'}

Checking updated balances...
Account 1 new balance: {'status': 'success', 'balance': 800.0, 'account_type': 'checking', 'holder_name': 'John Doe'}
Account 2 new balance: {'status': 'success', 'balance': 700.0, 'account_type': 'savings', 'holder_name': 'Jane Doe'}

Getting transaction history...
Transaction history: {'status': 'success', 'transactions': [...], 'count': 1}

Testing error cases...
Invalid amount transfer: {'status': 'error', 'message': 'Invalid transfer amount'}
Insufficient funds transfer: {'status': 'error', 'message': 'Insufficient funds'}
Invalid account balance check: {'status': 'error', 'message': 'Account not found'}
```

## Validation Checklist

- [ ] Cosmos DB containers created successfully
- [ ] Account creation works with validation
- [ ] Balance queries return accurate information
- [ ] Transfers process with proper validation
- [ ] Transaction history is properly recorded
- [ ] Error handling works for all cases
- [ ] Decimal amounts are properly handled

## Common Issues and Solutions

1. Cosmos DB Operations:

   - Check connection strings
   - Verify container creation
   - Monitor request units consumption

2. Transaction Processing:

   - Verify concurrent operation handling
   - Check error handling for insufficient funds
   - Validate account existence before transfers

3. Data Consistency:
   - Monitor balance updates
   - Verify transaction records
   - Check timestamp ordering

## Next Steps

In Lab 4, we'll:

- Implement vector search for products
- Add semantic caching
- Create product recommendation logic

Ah yes, thank you! Let me revise Lab 4 using Azure Cosmos DB NoSQL API's vector search capabilities correctly:

# Lab 4: Implementing Vector Search for Banking Products

## Overview

In this lab, you'll implement vector search for banking products using Azure Cosmos DB NoSQL API's vector search capabilities to provide intelligent product recommendations.

## Learning Objectives

- Enable vector search in Azure Cosmos DB NoSQL
- Set up vector-enabled container with proper indexing
- Implement product embeddings using Azure OpenAI
- Create semantic search for banking products

### Step 1: Enable Vector Search

First, enable vector search in your Azure Cosmos DB account:

```bash
az cosmosdb update \
     --resource-group <resource-group-name> \
     --name <account-name> \
     --capabilities EnableNoSQLVectorSearch
```

### Step 2: Update Cosmos DB Configuration

Update `src/app/services/azure_cosmos_db.py`:

```python
from azure.cosmos.aio import CosmosClient
from azure.cosmos import PartitionKey

class CosmosDB:
    async def initialize(self):
        try:
            # Existing containers remain the same
            await self.database.create_container_if_not_exists(
                id="accounts",
                partition_key=PartitionKey(path="/account_id")
            )
            await self.database.create_container_if_not_exists(
                id="transactions",
                partition_key=PartitionKey(path="/transaction_id")
            )

            # Create products container with vector search
            vector_policy = {
                "vectorEmbeddings": [
                    {
                        "path": "/embedding",
                        "dataType": "float32",
                        "distanceFunction": "cosine",
                        "dimensions": 1536
                    }
                ]
            }

            indexing_policy = {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [
                    {
                        "path": "/*"
                    }
                ],
                "excludedPaths": [
                    {
                        "path": "/_etag/?"
                    },
                    {
                        "path": "/embedding/*"
                    }
                ],
                "vectorIndexes": [
                    {
                        "path": "/embedding",
                        "type": "diskANN"
                    }
                ]
            }

            await self.database.create_container_if_not_exists(
                id="products",
                partition_key=PartitionKey(path="/category"),
                indexing_policy=indexing_policy,
                vector_policy=vector_policy
            )

            self.accounts = self.database.get_container_client("accounts")
            self.transactions = self.database.get_container_client("transactions")
            self.products = self.database.get_container_client("products")

            return True
        except Exception as e:
            print(f"Error initializing Cosmos DB: {str(e)}")
            return False
```

We need to implement the following later:

1. Product operations implementation with vector search
2. Azure OpenAI embeddings integration
3. Updated product tools with vector search
4. Testing framework

# Lab 5: Multi-tenant Banking API Implementation

## Overview

In this lab, you'll create a FastAPI application that supports multi-tenancy for the banking agents system. The API will handle session management, message processing, and agent coordination.

## Part 1: Basic Setup and Models

```python
import uuid
from azure.cosmos.exceptions import CosmosHttpResponseError
from fastapi import FastAPI, Depends, HTTPException, Body
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel
from typing import List
from langgraph_checkpoint_cosmosdb import CosmosDBSaver
from langgraph.graph.state import CompiledStateGraph
from starlette.middleware.cors import CORSMiddleware
from src.app.banking_agents import graph, checkpointer
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

endpointTitle = "ChatEndpoints"

# FastAPI app initialization
app = FastAPI(title="Cosmos DB Multi-Agent Banking API", openapi_url="/cosmos-multi-agent-api.json")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class DebugLog(BaseModel):
    id: str
    sessionId: str
    tenantId: str
    userId: str
    details: str

class Session(BaseModel):
    id: str
    type: str = "session"
    sessionId: str
    tenantId: str
    userId: str
    tokensUsed: int = 0
    name: str
    messages: List

class MessageModel(BaseModel):
    id: str
    type: str
    sessionId: str
    tenantId: str
    userId: str
    timeStamp: str
    sender: str
    senderRole: str
    text: str
    debugLogId: str
    tokensUsed: int
    rating: bool
    completionPromptId: str

def get_compiled_graph():
    return graph
```

## Part 2: Session Management

```python
def create_thread(tenantId: str, userId: str):
    """Creates a new chat session."""
    sessionId = str(uuid.uuid4())
    name = "John Doe"  # Default name, should be fetched from user profile in production
    age = 30
    address = "123 Main St"
    activeAgent = "unknown"
    ChatName = "New Chat"
    messages = []

    # Create session data in Cosmos DB
    update_userdata_container({
        "id": sessionId,
        "tenantId": tenantId,
        "userId": userId,
        "sessionId": sessionId,
        "name": name,
        "age": age,
        "address": address,
        "activeAgent": activeAgent,
        "ChatName": ChatName,
        "messages": messages
    })

    return Session(
        id=sessionId,
        sessionId=sessionId,
        tenantId=tenantId,
        userId=userId,
        name=name,
        messages=messages
    )

@app.post("/tenant/{tenantId}/user/{userId}/sessions", tags=[endpointTitle], response_model=Session)
def create_chat_session(tenantId: str, userId: str):
    """Creates a new chat session for a user."""
    return create_thread(tenantId, userId)

@app.get("/tenant/{tenantId}/user/{userId}/sessions/{sessionId}/messages",
         description="Retrieves messages from the sessionId", tags=[endpointTitle],
         response_model=List[MessageModel])
def get_chat_session(tenantId: str, userId: str, sessionId: str):
    """Retrieves all messages for a specific session."""
    return _fetch_messages_for_session(sessionId, tenantId, userId)
```

## Part 3: Message Processing

```python
def _fetch_messages_for_session(sessionId: str, tenantId: str, userId: str) -> List[MessageModel]:
    """Fetches messages for a session from the checkpoint store."""
    messages = []
    config = {
        "configurable": {
            "thread_id": sessionId,
            "checkpoint_ns": ""
        }
    }

    logging.debug(f"Fetching messages for sessionId: {sessionId} with config: {config}")
    checkpoints = list(checkpointer.list(config))
    logging.debug(f"Number of checkpoints retrieved: {len(checkpoints)}")

    if checkpoints:
        last_checkpoint = checkpoints[-1]
        for key, value in last_checkpoint.checkpoint.items():
            if key == "channel_values" and "messages" in value:
                messages.extend(value["messages"])

    # Find the last relevant message sequence
    selected_human_index = None
    for i in range(len(messages) - 1):
        if isinstance(messages[i], HumanMessage) and not isinstance(messages[i + 1], HumanMessage):
            selected_human_index = i
            break

    messages = messages[selected_human_index:] if selected_human_index is not None else []

    # Convert messages to MessageModel format
    return [
        MessageModel(
            id=str(uuid.uuid4()),
            type="ai_response",
            sessionId=sessionId,
            tenantId=tenantId,
            userId=userId,
            timeStamp=msg.response_metadata.get("timestamp", "") if hasattr(msg, "response_metadata") else "",
            sender="user" if isinstance(msg, HumanMessage) else "assistant",
            senderRole="user" if isinstance(msg, HumanMessage) else "agent",
            text=msg.content if hasattr(msg, "content") else msg.get("content", ""),
            debugLogId=str(uuid.uuid4()),
            tokensUsed=msg.response_metadata.get("token_usage", {}).get("total_tokens", 0) if hasattr(msg, "response_metadata") else 0,
            rating=True,
            completionPromptId=""
        )
        for msg in messages
        if msg.content
    ]
```

## Part 4: Chat Completion and Agent Routing

```python
@app.post("/tenant/{tenantId}/user/{userId}/sessions/{sessionId}/completion",
          tags=[endpointTitle], response_model=List[MessageModel])
def get_chat_completion(
    tenantId: str,
    userId: str,
    sessionId: str,
    request_body: str = Body(..., media_type="application/json"),
    workflow: CompiledStateGraph = Depends(get_compiled_graph)
):
    """Processes a new message and returns agent responses."""
    if not request_body.strip():
        raise HTTPException(status_code=400, detail="Request body cannot be empty")

    # Retrieve last checkpoint
    config = {"configurable": {"thread_id": sessionId, "checkpoint_ns": ""}}
    checkpoints = list(checkpointer.list(config))

    if not checkpoints:
        # No previous state, start fresh
        new_state = {
            "messages": [{"role": "user", "content": request_body}]
        }
        response_data = workflow.invoke(new_state, config, stream_mode="updates")
    else:
        # Resume from last checkpoint
        last_checkpoint = checkpoints[-1]
        last_state = last_checkpoint.checkpoint

        if "messages" not in last_state:
            last_state["messages"] = []

        last_state["messages"].append({"role": "user", "content": request_body})

        # Extract last active agent
        last_active_agent = "coordinator_agent"
        if "channel_versions" in last_state:
            for key in reversed(last_state["channel_versions"].keys()):
                if key.startswith("branch:") and "__self__:human" in key:
                    last_active_agent = key.split(":")[1]
                    break

        print(f"Resuming execution from last active agent: {last_active_agent}")
        last_state["langgraph_triggers"] = [f"resume:{last_active_agent}"]

        response_data = workflow.invoke(
            last_state,
            config,
            stream_mode="updates"
        )

    return extract_relevant_messages(response_data, tenantId, userId, sessionId)
```

## Part 5: Session Management and Cleanup

```python
def delete_all_thread_records(cosmos_saver: CosmosDBSaver, thread_id: str) -> None:
    """Deletes all records related to a thread from Cosmos DB."""
    query = "SELECT DISTINCT c.partition_key FROM c WHERE CONTAINS(c.partition_key, @thread_id)"
    parameters = [{"name": "@thread_id", "value": thread_id}]

    partition_keys = list(cosmos_saver.container.query_items(
        query=query, parameters=parameters, enable_cross_partition_query=True
    ))

    if not partition_keys:
        print(f"No records found for thread: {thread_id}")
        return

    print(f"Found {len(partition_keys)} partition keys related to the thread.")

    for partition in partition_keys:
        partition_key = partition["partition_key"]
        record_query = "SELECT c.id FROM c WHERE c.partition_key=@partition_key"
        record_parameters = [{"name": "@partition_key", "value": partition_key}]

        records = list(cosmos_saver.container.query_items(
            query=record_query, parameters=record_parameters, enable_cross_partition_query=True
        ))

        for record in records:
            try:
                cosmos_saver.container.delete_item(record["id"], partition_key=partition_key)
                print(f"Deleted record: {record['id']} from partition: {partition_key}")
            except CosmosHttpResponseError as e:
                print(f"Error deleting record {record['id']} (HTTP {e.status_code}): {e.message}")

@app.delete("/tenant/{tenantId}/user/{userId}/sessions/{sessionId}", tags=[endpointTitle])
def delete_chat_session(tenantId: str, userId: str, sessionId: str):
    """Deletes a chat session and all related messages."""
    delete_userdata_item(tenantId, userId, sessionId)
    delete_all_thread_records(checkpointer, sessionId)
    return {"message": "Session deleted successfully"}
```

## Part 6: Additional Utility Endpoints

```python
@app.get("/status", tags=[endpointTitle])
def get_service_status():
    """Gets the service status."""
    return "CosmosDBService: initializing"

@app.post("/tenant/{tenantId}/user/{userId}/sessions/{sessionId}/message/{messageId}/rate",
          tags=[endpointTitle], response_model=MessageModel)
def rate_message(tenantId: str, userId: str, sessionId: str, messageId: str, rating: bool):
    """Rates a message (placeholder for future implementation)."""
    return {
        "id": messageId,
        "type": "ai_response",
        "sessionId": sessionId,
        "tenantId": tenantId,
        "userId": userId,
        "timeStamp": "2023-01-01T00:00:00Z",
        "sender": "assistant",
        "senderRole": "agent",
        "text": "This is a rated message",
        "debugLogId": str(uuid.uuid4()),
        "tokensUsed": 0,
        "rating": rating,
        "completionPromptId": ""
    }
```

## Testing

Create a test script `test/test_api.py`:

```python
import requests
import uuid

BASE_URL = "http://localhost:8000"
TENANT_ID = "test-tenant"
USER_ID = "test-user"

def test_api():
    # Create session
    session_response = requests.post(
        f"{BASE_URL}/tenant/{TENANT_ID}/user/{USER_ID}/sessions"
    )
    session_id = session_response.json()["sessionId"]
    print(f"Created session: {session_id}")

    # Send message
    message = "I want to open a new account"
    completion_response = requests.post(
        f"{BASE_URL}/tenant/{TENANT_ID}/user/{USER_ID}/sessions/{session_id}/completion",
        json=message
    )
    print(f"Bot response: {completion_response.json()}")

    # Get messages
    messages_response = requests.get(
        f"{BASE_URL}/tenant/{TENANT_ID}/user/{USER_ID}/sessions/{session_id}/messages"
    )
    print(f"Session messages: {messages_response.json()}")

if __name__ == "__main__":
    test_api()
```

## Running the Application

1. Start the API server:

```bash
uvicorn src.app.banking_agents_api:app --reload
```

2. Run the test script:

```bash
python test/test_api.py
```

3. Access the Swagger documentation:

```
http://localhost:8000/docs
```
