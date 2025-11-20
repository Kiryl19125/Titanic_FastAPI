import uvicorn
from fastapi import FastAPI, HTTPException
from models.data import Data
from libs.model import make_prediction, DataFrame

MODEL_PATH = "ml_models/model.h5"

app = FastAPI()

@app.get("/", tags=["intro"])
async def index():
    return {"massege" : "Welcome to Titanic API"}

@app.post("/prediction", tags=["prediction"], status_code=200)
async def prediction(data: Data):
    try:
        data_frame = DataFrame(sex=data.sex, p_class=data.p_class, embark=data.embark,
                           age=data.age, sibsp=data.sibsp, parch=data.parch, fare=data.fare)
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))

    responce = make_prediction(model_path=MODEL_PATH, data_frame = data_frame)
    return responce

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)