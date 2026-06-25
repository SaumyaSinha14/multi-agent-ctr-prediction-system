import joblib


class EvaluationAgent:

    def evaluate_metrics(self, metrics, models):

        lr_accuracy = metrics["lr_accuracy"]
        xgb_accuracy = metrics["xgb_accuracy"]

        lr_precision = metrics["lr_precision"]
        xgb_precision = metrics["xgb_precision"]

        lr_recall = metrics["lr_recall"]
        xgb_recall = metrics["xgb_recall"]


        # Business decision logic

        if xgb_precision > lr_precision:

            final_model = "XGBoost"

            reason = "Better balanced performance"

            model_to_save = models["xgb_model"]

        else:

            final_model = "Logistic Regression"

            reason = "Higher precision"

            model_to_save = models["lr_model"]


        # Save final selected model

        joblib.dump(

            model_to_save,

            "models/best_model.pkl"

        )

        print("Best model saved successfully in models folder.")


        return {

            "final_model": final_model,

            "reason": reason
        }