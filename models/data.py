from pydantic import BaseModel

class Data(BaseModel):
    sex: str
    p_class: str
    embark: str
    age: int
    sibsp: int
    parch: int
    fare: int