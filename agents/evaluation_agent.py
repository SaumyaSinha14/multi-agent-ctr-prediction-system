class EvaluationAgent:

    def evaluate_metrics(self, metrics):

        lr_accuracy = metrics["lr_accuracy"]
        xgb_accuracy = metrics["xgb_accuracy"]

        lr_precision = metrics["lr_precision"]
        xgb_precision = metrics["xgb_precision"]

        lr_recall = metrics["lr_recall"]
        xgb_recall = metrics["xgb_recall"]


        # Business decision logic

        if lr_precision > xgb_precision:

            final_model = "Logistic Regression"

            reason = "Higher precision means fewer false positives."

        elif xgb_accuracy > lr_accuracy:

            final_model = "XGBoost"

            reason = "Higher accuracy and better general performance."

        else:

            final_model = "XGBoost"

            reason = "Better balanced performance."


        return {

            "final_model": final_model,

            "reason": reason
        }