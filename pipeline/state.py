from typing import TypedDict
import pandas as pd

class AgentState(TypedDict):

    dataframe: object

    metrics: dict

    models: dict

    final_model: str

    current_agent: str

    logs: list

