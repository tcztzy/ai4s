import json
from pathlib import Path
from typing import Annotated, Literal, TypeVar, cast, overload

import instructor
import typer
from anthropic import Anthropic, AnthropicBedrock, AnthropicVertex
from anthropic.types.message_param import MessageParam
from jinja2 import Template
from loguru import logger
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import BaseModel, Field, RootModel, TypeAdapter, ValidationError

from ai_scientist.llm import create_client

app = typer.Typer()

T = TypeVar("T")


class Idea(BaseModel, use_attribute_docstrings=True):
    Name: str
    """A shortened descriptor of the idea. Lowercase, no spaces, underscores allowed."""
    Title: str
    """A title for the idea, will be used for the report writing."""
    Experiment: str
    """An outline of the implementation. E.g. which functions need to be added or modified, how results will be obtained, ..."""
    Interestingness: Annotated[int, Field(ge=1, le=10)]
    """A rating from 1 to 10 (lowest to highest)."""
    Feasibility: Annotated[int, Field(ge=1, le=10)]
    """A rating from 1 to 10 (lowest to highest)."""
    Novelty: Annotated[int, Field(ge=1, le=10)]
    """A rating from 1 to 10 (lowest to highest)."""


idea_first_prompt = """{{task_description}}
<experiment.py>
{{code}}
</experiment.py>

{% if prev_ideas_string %}
Here are the ideas that you have already generated:

'''
{{prev_ideas_string}}
'''
{% endif %}

Come up with the next impactful and creative idea for research experiments and directions you can feasibly investigate with the code provided.
Note that you will not have access to any additional resources or datasets.
Make sure any idea is not overfit the specific training dataset or model, and has wider significance.

In <think>, first briefly discuss your intuitions and motivations for the idea.
Detail your high-level plan, necessary design choices and ideal outcomes of the experiments.
Justify how the idea is different from the existing ones.

Be cautious and realistic on your ratings.
{% if num_reflections > 1 %}
You will have {{num_reflections}} rounds to iterate on the idea, but do not need to use them all.
{% endif %}
"""

idea_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the quality, novelty, and feasibility of the idea you just created.
Include any other factors that you think are important in evaluating the idea.
Ensure the idea is clear and concise, and the JSON is the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your idea.
Stick to the spirit of the original idea unless there are glaring issues.

If there is nothing to improve, simply response "I am done".
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""

MAX_NUM_TOKENS = 4096


@overload
def get_response_from_llm(
    user_query: str,
    client: OpenAI | Anthropic | AnthropicBedrock | AnthropicVertex,
    model: str,
    system_message: str,
    messages: list[ChatCompletionMessageParam] | list[MessageParam],
    response_model: type[T],
    temperature: float = 0.75,
) -> tuple[T, list]: ...


@overload
def get_response_from_llm(
    user_query: str,
    client: OpenAI | Anthropic | AnthropicBedrock | AnthropicVertex,
    model: str,
    system_message: str,
    messages: list[ChatCompletionMessageParam] | list[MessageParam],
    response_model: None = None,
    temperature: float = 0.75,
) -> tuple[str, list]: ...


def get_response_from_llm(
    user_query: str,
    client: OpenAI | Anthropic | AnthropicBedrock | AnthropicVertex,
    model: str,
    system_message: str,
    messages: list[ChatCompletionMessageParam] | list[MessageParam],
    response_model: type[T] | None = None,
    temperature: float = 0.75,
) -> tuple[str | T, list]:
    if messages is None:
        messages = []

    match client:
        case OpenAI():
            messages = cast(list[ChatCompletionMessageParam], messages)
            messages = messages + [{"role": "user", "content": user_query}]
            is_o1 = model.startswith("o1") or model.startswith("o3")
            role = "user" if is_o1 else "system"
            params: CompletionCreateParamsBase = {
                "model": model,
                "messages": [
                    {"role": role, "content": system_message},
                    *messages,
                ],
                "n": 1,
            }
            match model:
                case str() if is_o1:
                    params["temperature"] = 1.0
                    params["max_completion_tokens"] = MAX_NUM_TOKENS
                case "deepseek-reasoner":
                    ...
                case _:
                    params["max_tokens"] = MAX_NUM_TOKENS
                    params["temperature"] = temperature
            if "gpt" in model or model.startswith("o1") or model.startswith("o3"):
                params["seed"] = 0  # best effort for reproducing
            if response_model is not None:
                if model == "deepseek-reasoner":
                    client = instructor.from_openai(
                        client, mode=instructor.Mode.MD_JSON
                    )
                else:
                    client = instructor.from_openai(client)
                content = cast(
                    T,
                    client.chat.completions.create(
                        response_model=response_model, **params
                    ),
                )
                messages = messages + [
                    {
                        "role": "assistant",
                        "content": cast(str, content.model_dump_json()),
                    }
                ]
            else:
                response = client.chat.completions.create(**params)
                content = response.choices[0].message.content or ""
                messages = messages + [{"role": "assistant", "content": content}]
        case Anthropic() | AnthropicBedrock() | AnthropicVertex():
            raise NotImplementedError("Anthropic is not supported.")
    logger.debug(f"""
******************** LLM START ********************
{"\n".join(f"{i}, {m['role']}: {m['content']}" for i, m in enumerate(messages))}
********************* LLM END *********************
""")

    return content, messages


def generate_ideas(
    base_dir: Path,
    client: OpenAI | Anthropic | AnthropicBedrock | AnthropicVertex,
    model: str,
    max_num_generations: int = 1,
    num_reflections: int = 1,
    force: bool = False,
) -> list[Idea]:
    if (ideas_json := base_dir / "ideas.json").exists() and not force:
        try:
            return TypeAdapter(list[Idea]).validate_json(ideas_json.read_text())
        except ValidationError:
            logger.warning("malformed ideas.json, regenerate ideas.")

    generated_ideas = []
    ideas = TypeAdapter(list[Idea]).validate_json(
        (base_dir / "seed_ideas.json").read_text()
    )
    code = (base_dir / "experiment.py").read_text()
    prompt = json.loads((base_dir / "prompt.json").read_text())
    idea_system_prompt = prompt["system"]

    for _ in range(max_num_generations):
        logger.info(f"Generating idea {_ + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join([idea.model_dump_json() for idea in ideas])

            msg_history = []
            logger.info(f"Iteration 1/{num_reflections}")
            new_idea, msg_history = get_response_from_llm(
                Template(idea_first_prompt).render(
                    task_description=prompt["task_description"],
                    code=code,
                    prev_ideas_string=prev_ideas_string,
                    num_reflections=num_reflections,
                ),
                client=client,
                model=model,
                system_message=idea_system_prompt,
                messages=msg_history,
                response_model=Idea,
            )

            # Iteratively improve task.
            if num_reflections > 1:
                for j in range(num_reflections - 1):
                    logger.info(f"Iteration {j + 2}/{num_reflections}")
                    _new_idea, msg_history = get_response_from_llm(
                        idea_reflection_prompt.format(
                            current_round=j + 2, num_reflections=num_reflections
                        ),
                        client=client,
                        model=model,
                        system_message=idea_system_prompt,
                        messages=msg_history,
                        response_model=RootModel[Literal["I am done"] | Idea],
                    )

                    if "I am done" == _new_idea.root:
                        logger.info(
                            f"Idea generation converged after {j + 2} iterations."
                        )
                        break
                    new_idea = cast(Idea, _new_idea.root)

            generated_ideas.append(new_idea)
        except Exception as e:
            logger.error("Failed to generate idea")
            logger.exception(e)
            continue

    ideas_json.write_bytes(TypeAdapter(list[Idea]).dump_json(ideas + generated_ideas))

    return ideas


@app.command()
def generate(
    experiment: Annotated[
        str, typer.Option(help="Experiment to run AI Scientist on.")
    ] = "nanoGPT",
    model: Annotated[
        str, typer.Option(help="Model to use for AI Scientist.")
    ] = "gpt-4o-2024-05-13",
    force: bool = False,
):
    """Generate AI scientist ideas."""
    # Create client
    client, client_model = create_client(model)

    base_dir = Path.cwd() / "templates" / experiment
    generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        force=force,
    )
