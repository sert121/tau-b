# Copyright Sierra

import json
import os
from pathlib import Path
from typing import Any, Optional, Union

from .rules import RULES
from .tools import ALL_TOOLS
from .wiki import WIKI
from ..base import Env
from ..user import UserStrategy

# Set the data folder path
FOLDER_PATH = Path(__file__).parent.parent.parent / "airline" / "data"

def load_data() -> dict[str, Any]:
    with open(os.path.join(FOLDER_PATH, "flights.json")) as f:
        flight_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "reservations.json")) as f:
        reservation_data = json.load(f)
    with open(os.path.join(FOLDER_PATH, "users.json")) as f:
        user_data = json.load(f)
    return {
        "flights": flight_data,
        "reservations": reservation_data,
        "users": user_data,
    }

class MockAirlineDomainEnv(Env):
    def __init__(
        self,
        user_strategy: Union[str, UserStrategy] = UserStrategy.LLM,
        user_model: str = "gpt-4o",
        user_provider: Optional[str] = None,
        task_split: str = "test",
        task_index: Optional[int] = None,
    ):
        match task_split:
            case "test":
                from .tasks_test import TASKS as tasks
            case _:
                raise ValueError(f"Unknown task split: {task_split}")
        super().__init__(
            data_load_func=load_data,
            tools=ALL_TOOLS,
            tasks=tasks,
            wiki=WIKI,
            rules=RULES,
            user_strategy=user_strategy,
            user_model=user_model,
            user_provider=user_provider,
            task_index=task_index,
        )
        self.terminate_tools = ["transfer_to_human_agents"]
