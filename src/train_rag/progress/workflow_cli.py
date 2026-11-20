"""`train-rag workflow ...`: start a workflow and report its steps from shell scripts."""

import argparse

from sqlmodel import Session

from train_rag.database import create_tables, get_engine
from train_rag.repository.workflow import (
    STATUSES,
    report_step_progress,
    set_step,
    start_workflow,
)


def add_workflow_commands(commands: "argparse._SubParsersAction[argparse.ArgumentParser]") -> None:
    """Add `workflow start`, `workflow step` and `workflow progress` to the CLI.

    Args:
        commands: The CLI's subcommands.
    """
    workflow = commands.add_parser(
        "workflow", help="show a multi-step job on the progress page, like CI/CD steps"
    )
    actions = workflow.add_subparsers(dest="action", required=True)
    start = actions.add_parser("start", help="record a workflow; prints its key")
    start.add_argument("name", help="what the workflow does")
    start.add_argument("steps", nargs="+", help="the step names, in order")
    step = actions.add_parser("step", help="set a step's status")
    step.add_argument("key", help="the workflow's key, from `workflow start`")
    step.add_argument("step", help="the step's name")
    step.add_argument("status", choices=STATUSES)
    step.add_argument("--note", help="extra line for the step")
    progress = actions.add_parser("progress", help="set a step's progress bar")
    progress.add_argument("key", help="the workflow's key")
    progress.add_argument("step", help="the step's name")
    progress.add_argument("done", type=int, help="items finished so far")
    progress.add_argument("total", type=int, help="items in all")
    progress.add_argument("--unit", default="steps", help="what the items are")
    progress.add_argument("--note", help="extra line for the step")


def run_workflow_command(args: argparse.Namespace) -> None:
    """Run one `workflow` subcommand.

    Args:
        args: The parsed command line.
    """
    engine = get_engine()
    create_tables(engine)
    with Session(engine) as session:
        if args.action == "start":
            print(start_workflow(session, args.name, args.steps))
        elif args.action == "step":
            set_step(session, args.key, args.step, args.status, args.note)
        elif args.action == "progress":
            report_step_progress(
                session, args.key, args.step, args.done, args.total, args.unit, args.note
            )
