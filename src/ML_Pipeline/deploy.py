import pandas as pd
from flask import Flask
from flask import request
import json
import Preprocess
import Predict
import Utils

app = Flask(__name__)

model_path = "../output/deep-ae-model"
ml_model, columns = Utils.load_model(model_path)


@app.post("/get_fraud_score")
async def get_license_status():
    items = json.loads(request.data)
    test_df = pd.DataFrame([items], columns=items.keys())
    processed_df = Preprocess.apply(test_df, is_train=False)
    prediction = Predict.init(processed_df, ml_model, columns)
    output = {"status": prediction}
    return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
