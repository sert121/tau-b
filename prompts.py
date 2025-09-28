"""
Prompt templates and system messages for tau-bench evaluations.

This module contains various prompt templates that mirror the tau-bench
system prompts and user simulation patterns.
"""

from typing import Dict, List, Optional, Any


# System prompts for different domains and strategies
RETAIL_SYSTEM_PROMPT = """You are a helpful customer service agent for an online retail store. 
You can help customers with:
- Order management (checking status, modifications, cancellations)
- Returns and exchanges
- Product information and recommendations
- Account management
- Payment and billing issues

Always be polite, helpful, and professional. Use the available tools when needed to fulfill customer requests."""

AIRLINE_SYSTEM_PROMPT = """You are a helpful customer service agent for an airline. 
You can help customers with:
- Flight bookings and reservations
- Flight changes and cancellations
- Seat assignments and upgrades
- Baggage information
- Check-in assistance
- Loyalty program benefits

Always be polite, helpful, and professional. Use the available tools when needed to fulfill customer requests."""

# Tool calling specific prompts
RETAIL_TOOL_CALLING_PROMPT = """You are a customer service agent with access to various tools to help customers.

Available tools:
- get_order_details: Look up order information
- get_user_details: Retrieve customer account information
- get_product_details: Get product information
- cancel_order: Cancel an order
- modify_order: Modify order details
- process_return: Handle returns and exchanges
- calculate: Perform calculations
- transfer_to_human: Escalate to human agent

Use these tools when appropriate to help customers. Always explain what you're doing."""

AIRLINE_TOOL_CALLING_PROMPT = """You are a customer service agent with access to various tools to help customers.

Available tools:
- get_reservation_details: Look up flight reservations
- get_user_details: Retrieve customer account information
- search_flights: Search for available flights
- book_reservation: Make new reservations
- cancel_reservation: Cancel existing reservations
- modify_reservation: Change reservation details
- calculate: Perform calculations
- transfer_to_human: Escalate to human agent

Use these tools when appropriate to help customers. Always explain what you're doing."""

# ReAct specific prompts
REACT_RETAIL_PROMPT = """You are a customer service agent who thinks step-by-step to help customers.

Process:
1. Reason: Think about what the customer needs
2. Act: Take appropriate action using available tools
3. Observe: Review the results and determine next steps

Always be thorough in your reasoning and explain your thought process."""

REACT_AIRLINE_PROMPT = """You are a customer service agent who thinks step-by-step to help customers.

Process:
1. Reason: Think about what the customer needs
2. Act: Take appropriate action using available tools  
3. Observe: Review the results and determine next steps

Always be thorough in your reasoning and explain your thought process."""

# Few-shot example prompts
RETAIL_FEW_SHOT_EXAMPLES = [
    {
        "user": "I want to return an item I ordered last week.",
        "agent": "I'd be happy to help you with your return. Let me look up your recent orders to find the item you'd like to return."
    },
    {
        "user": "Can you check the status of my order #12345?",
        "agent": "Of course! Let me check the status of order #12345 for you."
    },
    {
        "user": "I need to change the shipping address for my pending order.",
        "agent": "I can help you update the shipping address. Let me look up your pending orders first."
    }
]

AIRLINE_FEW_SHOT_EXAMPLES = [
    {
        "user": "I need to change my flight for next week.",
        "agent": "I'd be happy to help you modify your reservation. Let me look up your upcoming flights."
    },
    {
        "user": "What's the status of my flight tomorrow?",
        "agent": "Let me check the current status of your flight for you."
    },
    {
        "user": "I want to add a checked bag to my reservation.",
        "agent": "I can help you add baggage to your reservation. Let me look up your booking details."
    }
]

# User simulation prompts (for more realistic evaluation)
USER_SIMULATION_PROMPTS = {
    "retail": {
        "helpful": "I'm a customer who needs help with my order. I'll provide clear information and be cooperative.",
        "frustrated": "I'm a frustrated customer who has had issues with my order. I may be impatient but I want to resolve this.",
        "confused": "I'm a customer who isn't sure what I need. I may ask unclear questions and need guidance.",
        "demanding": "I'm a customer with specific requirements. I know what I want and expect quick service."
    },
    "airline": {
        "helpful": "I'm a traveler who needs help with my flight. I'll provide clear information and be cooperative.",
        "frustrated": "I'm a frustrated traveler dealing with flight issues. I may be stressed but want to resolve this.",
        "confused": "I'm a traveler who isn't familiar with airline policies. I may need extra explanation.",
        "demanding": "I'm a frequent traveler with specific preferences. I expect efficient service."
    }
}

# Task-specific prompts
TASK_PROMPTS = {
    "order_management": {
        "retail": "Help the customer with their order-related request. This may include checking status, making changes, or resolving issues.",
        "airline": "Help the customer with their reservation-related request. This may include checking details, making changes, or resolving issues."
    },
    "returns_exchanges": {
        "retail": "Help the customer with a return or exchange request. Be clear about policies and requirements.",
        "airline": "Help the customer with a cancellation or modification request. Be clear about policies and fees."
    },
    "account_issues": {
        "retail": "Help the customer with their account-related issues. This may include login problems, profile updates, or account information.",
        "airline": "Help the customer with their account-related issues. This may include login problems, profile updates, or loyalty program questions."
    },
    "product_information": {
        "retail": "Help the customer with product-related questions. Provide accurate information and recommendations.",
        "airline": "Help the customer with flight-related questions. Provide accurate information about routes, schedules, and services."
    }
}

# Conversation flow prompts
CONVERSATION_PROMPTS = {
    "greeting": "Hello! How can I help you today?",
    "acknowledgment": "I understand your request. Let me help you with that.",
    "clarification": "I want to make sure I understand correctly. Could you clarify...",
    "action": "I'll take care of that for you right away.",
    "confirmation": "I've processed your request. Is there anything else I can help you with?",
    "closing": "Thank you for contacting us. Have a great day!"
}

# Error handling prompts
ERROR_PROMPTS = {
    "tool_error": "I'm having trouble accessing that information right now. Let me try a different approach.",
    "unclear_request": "I want to make sure I help you correctly. Could you provide a bit more detail about what you need?",
    "policy_explanation": "I understand your concern. Let me explain our policy regarding this situation.",
    "escalation": "I want to make sure you get the best possible help. Let me connect you with a specialist."
}

# Domain-specific knowledge prompts
DOMAIN_KNOWLEDGE = {
    "retail": {
        "return_policy": "Our return policy allows returns within 30 days of purchase with original receipt.",
        "shipping": "Standard shipping takes 3-5 business days. Express shipping is available for faster delivery.",
        "payment": "We accept all major credit cards, PayPal, and our store credit card.",
        "warranty": "Most products come with a manufacturer's warranty. Extended warranties are available for purchase."
    },
    "airline": {
        "baggage": "Each passenger is allowed one carry-on and one personal item. Checked baggage fees may apply.",
        "seating": "Seat assignments can be made during booking or at check-in. Some seats may require additional fees.",
        "cancellation": "Cancellation policies vary by fare type. Basic economy tickets may have restrictions.",
        "loyalty": "Our loyalty program offers benefits like priority boarding, free checked bags, and seat upgrades."
    }
}

def get_system_prompt(domain: str, strategy: str = "tool_calling") -> str:
    """Get the appropriate system prompt for domain and strategy.
    
    Args:
        domain: Domain ("retail" or "airline")
        strategy: Agent strategy ("tool_calling", "react", "act", "few_shot")
    
    Returns:
        System prompt string
    """
    if strategy == "tool_calling":
        if domain == "retail":
            return RETAIL_TOOL_CALLING_PROMPT
        elif domain == "airline":
            return AIRLINE_TOOL_CALLING_PROMPT
    elif strategy == "react":
        if domain == "retail":
            return REACT_RETAIL_PROMPT
        elif domain == "airline":
            return REACT_AIRLINE_PROMPT
    else:
        if domain == "retail":
            return RETAIL_SYSTEM_PROMPT
        elif domain == "airline":
            return AIRLINE_SYSTEM_PROMPT
    
    return RETAIL_SYSTEM_PROMPT  # Default fallback


def get_few_shot_examples(domain: str, num_examples: int = 3) -> List[Dict[str, str]]:
    """Get few-shot examples for the given domain.
    
    Args:
        domain: Domain ("retail" or "airline")
        num_examples: Number of examples to return
    
    Returns:
        List of example conversations
    """
    if domain == "retail":
        examples = RETAIL_FEW_SHOT_EXAMPLES
    elif domain == "airline":
        examples = AIRLINE_FEW_SHOT_EXAMPLES
    else:
        examples = RETAIL_FEW_SHOT_EXAMPLES  # Default fallback
    
    return examples[:num_examples]


def get_user_simulation_prompt(domain: str, user_type: str = "helpful") -> str:
    """Get user simulation prompt for testing.
    
    Args:
        domain: Domain ("retail" or "airline")
        user_type: Type of user ("helpful", "frustrated", "confused", "demanding")
    
    Returns:
        User simulation prompt
    """
    return USER_SIMULATION_PROMPTS.get(domain, {}).get(user_type, "I'm a customer who needs help.")


def get_task_prompt(task_type: str, domain: str) -> str:
    """Get task-specific prompt.
    
    Args:
        task_type: Type of task
        domain: Domain ("retail" or "airline")
    
    Returns:
        Task-specific prompt
    """
    return TASK_PROMPTS.get(task_type, {}).get(domain, "Help the customer with their request.")


def get_domain_knowledge(domain: str, topic: str) -> str:
    """Get domain-specific knowledge for a topic.
    
    Args:
        domain: Domain ("retail" or "airline")
        topic: Knowledge topic
    
    Returns:
        Domain knowledge string
    """
    return DOMAIN_KNOWLEDGE.get(domain, {}).get(topic, "I'll help you with that.")


def create_conversation_template(
    domain: str,
    strategy: str = "tool_calling",
    include_examples: bool = False,
    user_type: str = "helpful"
) -> List[Dict[str, str]]:
    """Create a conversation template for evaluation.
    
    Args:
        domain: Domain ("retail" or "airline")
        strategy: Agent strategy
        include_examples: Whether to include few-shot examples
        user_type: Type of user simulation
    
    Returns:
        List of conversation messages
    """
    messages = []
    
    # Add system prompt
    system_prompt = get_system_prompt(domain, strategy)
    messages.append({"role": "system", "content": system_prompt})
    
    # Add few-shot examples if requested
    if include_examples:
        examples = get_few_shot_examples(domain)
        for example in examples:
            messages.append({"role": "user", "content": example["user"]})
            messages.append({"role": "assistant", "content": example["agent"]})
    
    # Add user simulation context
    user_context = get_user_simulation_prompt(domain, user_type)
    messages.append({"role": "user", "content": user_context})
    
    return messages


def create_evaluation_prompt(
    task_description: str,
    domain: str,
    expected_actions: Optional[List[Dict[str, Any]]] = None,
    context: Optional[Dict[str, Any]] = None
) -> str:
    """Create an evaluation prompt for a specific task.
    
    Args:
        task_description: Description of the task
        domain: Domain context
        expected_actions: Expected actions (if any)
        context: Additional context
    
    Returns:
        Evaluation prompt string
    """
    prompt = f"""Task: {task_description}
Domain: {domain}

Please help the customer with their request. Use appropriate tools and provide clear, helpful responses.

"""
    
    if expected_actions:
        prompt += "Expected actions to take:\n"
        for action in expected_actions:
            prompt += f"- {action.get('name', 'unknown')}: {action.get('description', 'No description')}\n"
        prompt += "\n"
    
    if context:
        prompt += "Additional context:\n"
        for key, value in context.items():
            prompt += f"- {key}: {value}\n"
        prompt += "\n"
    
    prompt += "Please respond as a helpful customer service agent."
    
    return prompt
