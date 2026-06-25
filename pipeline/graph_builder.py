import sys
import os

#print("Current file:", __file__)
#print("Current working dir:", os.getcwd())

project_root = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

#print("Detected project root:", project_root)

sys.path.insert(0, project_root)

#print("Python import path:", sys.path)

from agents.data_agent import DataAgent


from langgraph.graph import StateGraph
from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from agents.user_behavior_agent import UserBehaviorAgent
from agents.prediction_agent import PredictionAgent
from pipeline.state import AgentState
from langgraph.graph import START, END
from agents.evaluation_agent import EvaluationAgent

data_agent = DataAgent("data/raw/ctr_dataset.csv"
)

feature_agent = FeatureAgent()

behavior_agent = UserBehaviorAgent()

prediction_agent = PredictionAgent()
evaluation_agent = EvaluationAgent()
def data_node(state):

    df = data_agent.load_data()

    return {

        "dataframe": df,

        "current_agent":"feature_agent"
    }

def feature_node(state):

    df = state["dataframe"]

    df = feature_agent.encode_categorical_columns(df)

    return {

        "dataframe": df,

        "current_agent": "user_behavior_agent"
    }

def user_behavior_node(state):

    df = state["dataframe"]

    df = behavior_agent.create_behavior_score(df)

    return {

        "dataframe": df,

        "current_agent": "prediction_agent"
    }


def prediction_node(state):

    df = state["dataframe"]

    result = prediction_agent.train_models(df)

    return {

        "metrics": result["metrics"],

        "models": result["models"],

        "current_agent": "evaluation_agent"
    }
    
def evaluation_node(state):

    metrics = state["metrics"]

    models = state["models"]

    result = evaluation_agent.evaluate_metrics(
        metrics,
        models
    )

    return {

        "final_model": result["final_model"],

        "logs": [

            f'Final Model Selected: {result["final_model"]}',

            result["reason"]

        ],

        "current_agent": "completed"
    }

workflow = StateGraph(AgentState)
workflow.add_node("data", data_node)
workflow.add_node("feature",feature_node)
workflow.add_node("behavior",user_behavior_node)
workflow.add_node("prediction",prediction_node)
workflow.add_node("evaluation", evaluation_node)

workflow.add_edge(START, "data")
workflow.add_edge("data", "feature")
workflow.add_edge("feature","behavior")
workflow.add_edge("behavior","prediction")
workflow.add_edge("prediction", "evaluation")
workflow.add_edge("evaluation", END)
app = workflow.compile()

initial_state = {

    "dataframe": None,

    "metrics": {},

    "best_model_name": None,

    "best_model_object": None,

    "current_agent": "not_running",

    "logs": []
}
result = app.invoke(initial_state)
print(result)
