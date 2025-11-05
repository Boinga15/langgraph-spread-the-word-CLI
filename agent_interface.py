# Imports
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolCall

from langgraph.func import entrypoint, task

from typing_extensions import TypedDict

# Setting Up Types
class HeadlineScore(TypedDict):
    """The scores of evaluations for a headline."""
    contextScore: int # The score of the context evaluation between 0 and 100.
    noContextScore: int # The score of the no-context evaluation between 0 and 100.
    interestScore: int # The score of the interest evaluation between 0 and 100.


# Construct LLMs
describerLLM = init_chat_model(
    model = "ollama:llama3.2",
    temperature = 0.5
)

evaluatorLLM = create_agent(
    model = "ollama:llama3.2",
    response_format = HeadlineScore
)


# Build Tasks
@task
def evaluate_with_context(nextHeadline: str, previousHeadlineResults):
    messages = [
        SystemMessage("You are a judge for the belivability of fake news headlines.\n\nThe user will provide you with a fake news headline. You are to judge how likely it is that people will accept it as true.\n\nYou should respond in a professional manner, and detail your reasoning.\n\nAlongside the provided fake news headline, the user will also provide you with previous headlines they have made, with both the headline title as well as the percentage of people who have generally accepted the headline to be factual. You should consider these headlines and their success rate when considering how believable this new headline would be.\n\nIf no previous headlines were given, or no previous headline links with this new headline, then judge the new headline by how believable it would be to someone who has no idea about anything in the field the news headline is talking about.\n\nIn your evaluation, you should attempt to rate the headline out of 100% on the basis of believability, where 0% means not believable at all, and 100% means completely believable.")
    ]

    for headline in previousHeadlineResults:
        messages.append(HumanMessage(f"Previous Headline: \"{headline[0]}\" - Success Rate: {headline[1]}%."))
    
    messages.append(HumanMessage(f"New Headline: \"{nextHeadline}\""))

    msg = describerLLM.invoke(messages)
    return msg.content


@task
def evaluate_without_context(nextHeadline: str):
    messages = [
        SystemMessage("You are a judge for the belivability of fake news headlines.\n\nThe user will provide you with a fake news headline. You are to judge how likely it is that people will accept it as true.\n\nYou should respond in a professional manner, and detail your reasoning.\n\nIn your evaluation, you should attempt to rate the headline out of 100% on the basis of believability, where 0% means not believable at all, and 100% means completely believable.")
    ]
    
    messages.append(HumanMessage(f"New Headline: \"{nextHeadline}\""))

    msg = describerLLM.invoke(messages)
    return msg.content


@task
def evaluate_interest(nextHeadline: str, previousHeadlineResults):
    messages = [
        SystemMessage("You are a judge for the possible interest of fake news headlines.\n\nThe user will provide you with a fake news headline. You are to judge how interesting the news article would be based on its headline.\n\nYou should respond in a professional manner, and detail your reasoning.\n\nAlongside the provided fake news headline, the user will also provide you with previous headlines they have made. You should consider these headlines and their success rate when considering how interesting this new headline would be.\n\nIn your evaluation, you should attempt to rate the headline out of 100% on the basis of believability, where 0% means not believable at all, and 100% means completely believable.")
    ]

    for headline in previousHeadlineResults:
        messages.append(HumanMessage(f"Previous Headline: \"{headline[0]}\" - Success Rate: {headline[1]}%."))
    
    messages.append(HumanMessage(f"New Headline: \"{nextHeadline}\""))

    msg = describerLLM.invoke(messages)

    return msg.content


@task
def evaluate_probability(contextEval, noContextEval, interestEval):
    """
    messages = [
        SystemMessage("You are a judge for the overall score of fake news headlines.\n\nThe user will provide you with an evaluation of a fake news headline with context, an evaluation of a fake news headline without context, and an evaluation of a fake news headline in terms of how interesting it is.\n\nYou have access to two tools to help you come up with a final score for the headline:" \
        "" \
        "- get_headline_score takes in the score for the context, no-context, and interest evaluation, and returns the final score of the headline." \
        "- get_headline_believability gets in the score for the context, no-context, and interest evaluation, and returns a decimal number representing the believability of the headline." \
        "" \
        "You should generate a score between 0 and 1 for the context evaluation, no-context evaluation, and interest evaluation, where 0 is the worst and 1 is the best. No score should be below 0, and no score should be below 1." \
        "" \
        "Your final response must be in the form [headline score]|[believability score]. You should use get_headline_score to generate the headline score, and you should use get_headline_believability to get the believability score. You should only respond in this format and add no further justification or reasoning to your response."),
        HumanMessage(f"Context Evaluation: {contextEval}"),
        HumanMessage(f"No Context Evaluation: {noContextEval}"),
        HumanMessage(f"Interest Evaluation: {interestEval}")
    ]

    msg = evaluatorLLM.invoke(messages)

    while True:
        if not msg.tool_calls:
            break

        # Execute Tools
        toolResultFutures = [
            call_tool(tool_call) for tool_call in msg.tool_calls
        ]

        toolResults = [future.result() for future in toolResultFutures]
        messages = add_messages(messages, [msg, *toolResults])
        msg = evaluatorLLM.invoke(messages)

    """

    messages = [
        SystemMessage("You are a judge for the overall score of fake news headlines.\n\nThe user will provide you with an evaluation of a fake news headline with context, an evaluation of a fake news headline without context, and an evaluation of a fake news headline in terms of how interesting it is.\n\n" \
        "" \
        "For each evaluation, you should decide on a score for each between 0 and 100, where 0 is bad and 100 is best. Your score should be based on the content of each evaluation, and the score should be representation of the overall opinion from the evaluation." \
        "" \
        "When deciding on a score for an evaluation, if it gives a final percentage, use that but ensure that their percentage lines up with the rest of the evaluation. Furthermore if they state something like 4 / 10 or 2 / 10, use that, but ensure to scale it up to a 0 to 100 scale, and evaluate the given number against the rest of the evaluation."),
        HumanMessage(f"Context Evaluation: {contextEval}"),
        HumanMessage(f"No Context Evaluation: {noContextEval}"),
        HumanMessage(f"Interest Evaluation: {interestEval}")
    ]
    msg = evaluatorLLM.invoke({"messages": messages})

    return msg["structured_response"]


# LLM Graph
@entrypoint()
def headline_workflow(inputData):
    headline = inputData["headline"]
    previousHeadlines = inputData["previous_headlines"]

    contextEval = evaluate_with_context(headline, previousHeadlines).result()
    noContextEval = evaluate_without_context(headline).result()
    interestEval = evaluate_interest(headline, previousHeadlines).result()

    return evaluate_probability(contextEval, noContextEval, interestEval).result()