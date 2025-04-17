import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time
from datetime import datetime
from enum import StrEnum
from typing import Annotated

import openai
import torch
import typer
from aider.coders import Coder
from aider.io import InputOutput
from aider.models import Model
from loguru import logger

from ai_scientist.generate_ideas import check_idea_novelty
from ai_scientist.llm import create_client
from ai_scientist.perform_experiments import perform_experiments
from ai_scientist.perform_review import load_paper, perform_improvement, perform_review
from ai_scientist.perform_writeup import generate_latex, perform_writeup

from . import idea
from .idea import generate_ideas

app = typer.Typer()
app.add_typer(idea.app, name="idea")

NUM_REFLECTIONS = 3


def get_available_gpus(gpu_ids: str) -> list[int]:
    return [int(gpu_id) for gpu_id in gpu_ids.split(",")]


def check_latex_dependencies():
    """
    Check if required LaTeX dependencies are installed on the system.
    Returns True if all dependencies are found, False otherwise.
    """
    for dep in ("pdflatex", "chktex"):
        if shutil.which(dep) is None:
            logger.error(f"Required LaTeX dependencies {dep} not found")
            return False

    return True


def worker(
    queue,
    base_dir,
    results_dir,
    model,
    client,
    client_model,
    writeup,
    improvement,
    gpu_id,
    engine,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logger.info(f"Worker {gpu_id} started.")
    while True:
        idea = queue.get()
        if idea is None:
            break
        success = do_idea(
            base_dir,
            results_dir,
            idea,
            model,
            client,
            client_model,
            writeup,
            improvement,
            engine,
            log_file=True,
        )
        logger.info(f"Completed idea: {idea['Name']}, Success: {success}")
    logger.info(f"Worker {gpu_id} finished.")


def do_idea(
    base_dir,
    results_dir,
    idea,
    model,
    client,
    client_model,
    writeup,
    improvement,
    engine,
    log_file=False,
):
    ## CREATE PROJECT FOLDER
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    idea_name = f"{timestamp}_{idea['Name']}"
    folder_name = osp.join(results_dir, idea_name)
    assert not osp.exists(folder_name), f"Folder {folder_name} already exists."
    destination_dir = folder_name
    shutil.copytree(base_dir, destination_dir, dirs_exist_ok=True)
    with open(osp.join(base_dir, "run_0", "final_info.json"), "r") as f:
        baseline_results = json.load(f)
    # Check if baseline_results is a dictionary before extracting means
    if isinstance(baseline_results, dict):
        baseline_results = {k: v["means"] for k, v in baseline_results.items()}
    exp_file = osp.join(folder_name, "experiment.py")
    vis_file = osp.join(folder_name, "plot.py")
    notes = osp.join(folder_name, "notes.txt")
    with open(notes, "w") as f:
        f.write(f"# Title: {idea['Title']}\n")
        f.write(f"# Experiment description: {idea['Experiment']}\n")
        f.write("## Run 0: Baseline\n")
        f.write(f"Results: {baseline_results}\n")
        f.write("Description: Baseline results.\n")
    if log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        log_path = osp.join(folder_name, "log.txt")
        log = open(log_path, "a")
        sys.stdout = log
        sys.stderr = log
    try:
        logger.info(f"*Starting idea: {idea_name}*")
        ## PERFORM EXPERIMENTS
        fnames = [exp_file, vis_file, notes]
        io = InputOutput(
            yes=True, chat_history_file=f"{folder_name}/{idea_name}_aider.txt"
        )
        if model == "deepseek-coder-v2-0724":
            main_model = Model("deepseek/deepseek-coder")
        elif model == "deepseek-reasoner":
            main_model = Model("deepseek/deepseek-reasoner")
        elif model == "llama3.1-405b":
            main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
        else:
            main_model = Model(model)
        coder = Coder.create(
            main_model=main_model,
            fnames=fnames,
            io=io,
            stream=False,
            use_git=False,
            edit_format="diff",
        )

        logger.info("*Starting Experiments*")
        try:
            success = perform_experiments(idea, folder_name, coder, baseline_results)
        except Exception as e:
            logger.error(f"Experiments failed for idea {idea_name}")
            logger.exception(e)
            return False

        if not success:
            logger.error(f"Experiments failed for idea {idea_name}")
            return False

        logger.info("*Starting Writeup*")
        ## PERFORM WRITEUP
        if writeup == "latex":
            writeup_file = osp.join(folder_name, "latex", "template.tex")
            fnames = [exp_file, writeup_file, notes]
            if model == "deepseek-coder-v2-0724":
                main_model = Model("deepseek/deepseek-coder")
            elif model == "deepseek-reasoner":
                main_model = Model("deepseek/deepseek-reasoner")
            elif model == "llama3.1-405b":
                main_model = Model("openrouter/meta-llama/llama-3.1-405b-instruct")
            else:
                main_model = Model(model)
            coder = Coder.create(
                main_model=main_model,
                fnames=fnames,
                io=io,
                stream=False,
                use_git=False,
                edit_format="diff",
            )
            try:
                perform_writeup(
                    idea, folder_name, coder, client, client_model, engine=engine
                )
            except Exception as e:
                logger.error("Failed to perform writeup")
                logger.exception(e)
                return False
            logger.info("Done writeup")
        else:
            raise ValueError(f"Writeup format {writeup} not supported.")

        logger.info("*Starting Review*")
        ## REVIEW PAPER
        if writeup == "latex":
            try:
                paper_text = load_paper(f"{folder_name}/{idea['Name']}.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review.txt"), "w") as f:
                    f.write(json.dumps(review, indent=4))
            except Exception as e:
                logger.error("Failed to perform review")
                logger.exception(e)
                return False

        ## IMPROVE WRITEUP
        if writeup == "latex" and improvement:
            logger.info("*Starting Improvement*")
            try:
                perform_improvement(review, coder)
                generate_latex(
                    coder, folder_name, f"{folder_name}/{idea['Name']}_improved.pdf"
                )
                paper_text = load_paper(f"{folder_name}/{idea['Name']}_improved.pdf")
                review = perform_review(
                    paper_text,
                    model="gpt-4o-2024-05-13",
                    client=openai.OpenAI(),
                    num_reflections=5,
                    num_fs_examples=1,
                    num_reviews_ensemble=5,
                    temperature=0.1,
                )
                # Store the review in separate review.txt file
                with open(osp.join(folder_name, "review_improved.txt"), "w") as f:
                    f.write(json.dumps(review))
            except Exception as e:
                logger.exception(f"Failed to perform improvement: {e}")
                return False
        return True
    except Exception as e:
        logger.error(f"Failed to evaluate idea {idea_name}")
        logger.exception(e)
        return False
    finally:
        logger.info("FINISHED IDEA")
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log.close()


class WriteUp(StrEnum):
    """Writeup formats."""

    latex = "latex"


class Engine(StrEnum):
    semanticscholar = "semanticscholar"
    openalex = "openalex"


@app.command()
def run(
    force_idea_generation: Annotated[
        bool,
        typer.Option(
            "--force-idea-generation/",
            help="Force idea generation",
        ),
    ] = False,
    skip_novelty_check: Annotated[
        bool,
        typer.Option(
            "--skip-novelty-check/",
            help="Skip novelty check and use existing ideas",
        ),
    ] = False,
    experiment: Annotated[
        str, typer.Option(help="Experiment to run AI Scientist on.")
    ] = "nanoGPT",
    model: Annotated[
        str, typer.Option(help="Model to use for AI Scientist.")
    ] = "claude-3-5-sonnet-20240620",
    writeup: Annotated[
        WriteUp,
        typer.Option(case_sensitive=False, help="What format to use for writeup"),
    ] = WriteUp.latex,
    parallel: Annotated[
        int,
        typer.Option(
            help="Number of parallel processes to run. 0 for sequential execution."
        ),
    ] = 0,
    improvement: Annotated[
        bool, typer.Option("--improvement/", help="Improve based on reviews.")
    ] = False,
    gpus: Annotated[
        list[int],
        typer.Option(
            parser=get_available_gpus,
            help="Comma-separated list of GPU IDs to use (e.g., '0,1,2'). If not specified, all available GPUs will be used.",
        ),
    ] = list(range(torch.cuda.device_count())),
    num_ideas: Annotated[int, typer.Option(help="Number of ideas generate")] = 50,
    engine: Annotated[
        Engine, typer.Option(help="Scholar engine to use.")
    ] = Engine.semanticscholar,
):
    """Run AI scientist experiments."""

    # Check available GPUs and adjust parallel processes if necessary
    if parallel > len(gpus):
        logger.warning(
            f"Requested {parallel} parallel processes, "
            f"but only {len(gpus)} GPUs available. "
            f"Adjusting to {len(gpus)}."
        )
        parallel = len(gpus)

    logger.info(f"Using GPUs: {gpus}")

    # Check LaTeX dependencies before proceeding
    if writeup == WriteUp.latex and not check_latex_dependencies():
        sys.exit(1)

    # Create client
    client, client_model = create_client(model)

    base_dir = osp.join("templates", experiment)
    results_dir = osp.join("results", experiment)
    ideas = generate_ideas(
        base_dir,
        client=client,
        model=client_model,
        max_num_generations=num_ideas,
        num_reflections=NUM_REFLECTIONS,
        force=force_idea_generation,
    )
    if not skip_novelty_check:
        ideas = check_idea_novelty(
            ideas,
            base_dir=base_dir,
            client=client,
            model=client_model,
            engine=engine.value,
        )

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    novel_ideas = [idea for idea in ideas if idea["novel"]]

    if parallel > 0:
        logger.info(f"Running {parallel} parallel processes")
        queue = multiprocessing.Queue()
        for idea in novel_ideas:
            queue.put(idea)

        processes = []
        for i in range(parallel):
            gpu_id = gpus[i % len(gpus)]
            p = multiprocessing.Process(
                target=worker,
                args=(
                    queue,
                    base_dir,
                    results_dir,
                    model,
                    client,
                    client_model,
                    writeup.value,
                    improvement,
                    gpu_id,
                    engine,
                ),
            )
            p.start()
            time.sleep(150)
            processes.append(p)

        # Signal workers to exit
        for _ in range(parallel):
            queue.put(None)

        for p in processes:
            p.join()

        logger.info("All parallel processes completed.")
    else:
        for idea in novel_ideas:
            logger.info(f"Processing idea: {idea['Name']}")
            try:
                success = do_idea(
                    base_dir,
                    results_dir,
                    idea,
                    model,
                    client,
                    client_model,
                    writeup.value,
                    improvement,
                    engine,
                )
                logger.info(f"Completed idea: {idea['Name']}, Success: {success}")
            except Exception as e:
                logger.error(f"Failed to evaluate idea {idea['Name']}")
                logger.exception(e)
    logger.info("All ideas evaluated.")


app()
