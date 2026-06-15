from typing import TypedDict
import pandas as pd

class AgentState(TypedDict):

    dataframe: pd.DataFrame

    metrics: dict

    final_model: str

    current_agent: str

    logs: list[str]

