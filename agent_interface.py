# Imports
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolCall
from langchain.tools import tool

from langgraph.func import entrypoint, task

# Construct LLMs
describerLLM = init_chat_model(
    model = "ollama:llama3.2",
    temperature = 0.5
)

evaluatorLLM = init_chat_model(
    model = "ollama:llama3.2",
    temperature = 0
)


# Build Tools
@tool
def get_headline_score(contextScore: float, noContextScore: float, interestScore: float):
    """
    Given a score for context, no context, and interest, calculate the final score of a headline.

    Args:
        contextScore: The context score of the headline.
        noContextScore: The no-context score of the headline.
        interestScore: The interest score of the headline.
    """

    finalScore = 1000 * (interestScore + 0.5) * (1 + contextScore - noContextScore)

    return finalScore


@tool
def get_headline_believability(contextScore: float, noContextScore: float, interestScore: float):
    """
    Given a score for context, no context, and interest, calculate the final believability of a headline.

    Args:
        contextScore: The context score of the headline.
        noContextScore: The no-context score of the headline.
        interestScore: The interest score of the headline.
    """

    finalScore = contextScore * 0.7 + noContextScore * 0.2 + interestScore * 0.1

    return finalScore


# Integrating Tools
tools = [get_headline_score, get_headline_believability]
toolsByName = {tool.name: tool for tool in tools}

@task
def call_tool(tool_call: ToolCall):
    tool = toolsByName[tool_call["name"]]
    return tool.invoke(tool_call)

evaluatorLLM = evaluatorLLM.bind_tools(tools)


# Build Tasks
@task
def evaluate_with_context(nextHeadline: str, previousHeadlineResults):
    messages = [
        SystemMessage("You are a judge for the belivability of fake news headlines.\n\nThe user will provide you with a fake news headline. You are to judge how likely it is that people will accept it as true.\n\nYou should respond in a professional manner, and detail your reasoning.\n\nAlongside the provided fake news headline, the user will also provide you with previous headlines they have made, with both the headline title as well as the percentage of people who have generally accepted the headline to be factual. You should consider these headlines and their success rate when considering how believable this new headline would be.")
    ]

    for headline in previousHeadlineResults:
        messages.append(HumanMessage(f"Previous Headline: \"{headline[0]}\" - Success Rate: {headline[1]}%."))
    
    messages.append(HumanMessage(f"New Headline: \"{nextHeadline}\""))

    msg = describerLLM.invoke(messages)
    return msg.content


@task
def evaluate_without_context(nextHeadline: str):
    messages = [
        SystemMessage("You are a judge for the belivability of fake news headlines.\n\nThe user will provide you with a fake news headline. You are to judge how likely it is that people will accept it as true.\n\nYou should respond in a professional manner, and detail your reasoning.")
    ]
    
    messages.append(HumanMessage(f"New Headline: \"{nextHeadline}\""))

    msg = describerLLM.invoke(messages)
    return msg.content


@task
def evaluate_interest(nextHeadline: str, previousHeadlineResults):
    messages = [
        SystemMessage("You are a judge for the possible interest of fake news headlines.\n\nThe user will provide you with a fake news headline. You are to judge how interesting the news article would be based on its headline.\n\nYou should respond in a professional manner, and detail your reasoning.\n\nAlongside the provided fake news headline, the user will also provide you with previous headlines they have made. You should consider these headlines and their success rate when considering how interesting this new headline would be.")
    ]

    for headline in previousHeadlineResults:
        messages.append(HumanMessage(f"Previous Headline: \"{headline[0]}\" - Success Rate: {headline[1]}%."))
    
    messages.append(HumanMessage(f"New Headline: \"{nextHeadline}\""))

    msg = describerLLM.invoke(messages)
    return msg.content


@task
def evaluate_probability(contextEval, noContextEval, interestEval):
    messages = [
        SystemMessage("You are a judge for the overall score of fake news headlines.\n\nThe user will provide you with an evaluation of a fake news headline with context, an evaluation of a fake news headline without context, and an evaluation of a fake news headline in terms of how interesting it is.\n\nYou should only respond with a single number between 0 and 1 giving a score for the headline based on these three evaluations. You should not provide any explanation for your final score, only respond with the score. A score of 0 is extremely bad, while a score of 1 is extremely good.\n\nHeadlines that score the best are those where the context evaluation is great, and the interest evaluation is great, however the no context evaluation is extremely bad.\n\nHeadlines that score worst have extremely good no context evaluations, but extremely bad context and interest evaluations."),
        HumanMessage(f"Context Evaluation: {contextEval}"),
        HumanMessage(f"No Context Evaluation: {noContextEval}"),
        HumanMessage(f"Interest Evaluation: {interestEval}")
    ]

    msg = evaluatorLLM.invoke(messages)
    return msg.content

# LLM Graph
@entrypoint()
def headline_workflow(inputData):
    headline = inputData["headline"]
    previousHeadlines = inputData["previous_headlines"]

    contextEval = evaluate_with_context(headline, previousHeadlines).result()
    noContextEval = evaluate_without_context(headline).result()
    interestEval = evaluate_interest(headline, previousHeadlines).result()

    return evaluate_probability(contextEval, noContextEval, interestEval)

for step in headline_workflow.stream({
    "headline": "Scientists claim that the idea that pigs used to have wings is 'baseless conjecture' despite new evidence.",
    "previous_headlines": [["Hidden bone structure in pigs reveal the possibility that pigs could grow wings.", 93], ["After great experimentation, scientists successfully create pig with small wings.", 75]]
}, stream_mode="updates"):
    print(step)
    print("\n")