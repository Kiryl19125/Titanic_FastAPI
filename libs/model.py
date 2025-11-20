import pickle

class DataFrame:

    _SEX_VALUE_MAPPER = {"Kobieta": 0, "Mezczycna": 1}
    _CLASS_VALUE_MAPPER = {"Pierwsza": 0, "Druga": 1, "Trzecia": 2}
    _EMBARK_VALUE_MAPPER = {"Cherbourg": 0, "Queenstown": 1, "Southapton": 2}

    _SEX_TYPES: list[str] = ["Kobieta", "Mezczycna"]
    _CLASS_TYPES: list[str] = ["Pierwsza", "Druga", "Trzecia"]
    _EMBARK_TYPES: list[str] = ["Cherbourg", "Queenstown", "Southapton"]

    _AGE_MIN: int = 1
    _AGE_MAX: int = 100
    _SIBSP_MIN: int = 0
    _SIBSP_MAX: int = 8
    _PARCH_MIN: int = 0
    _PARCH_MAX: int = 6
    _FARE_MIN: int = 0
    _FARE_MAX: int = 500

    def __init__(self, sex: str, p_class: str, embark: str, age: int, sibsp: int, parch: int, fare: int):
        self.sex = self._validate_sex(sex)
        self.p_class = self._validate_class(p_class)
        self.embark = self._validate_embark(embark)
        self.age = self._validate_age(age)
        self.sibsp = self._validate_sibsp(sibsp)
        self.parch = self._validate_parch(parch)
        self.fare = self._validate_fare(fare)

    def _validate_sex(self, sex: str) -> str:
        if sex not in self._SEX_TYPES:
            raise ValueError(f"sex must be: {self._SEX_TYPES}")

        return sex

    def _validate_class(self, p_class: str) -> str:
        if p_class not in self._CLASS_TYPES:
            raise ValueError(f"class must be: {self._CLASS_TYPES}")

        return p_class

    def _validate_embark(self, embark: str) -> str:
        if embark not in self._EMBARK_TYPES:
            raise ValueError(f"embark must be: {self._EMBARK_TYPES}")

        return embark

    def _validate_age(self, age: int) -> int:
        if age < 1 or age > 100:
            raise ValueError(f"Age must be between {self._AGE_MIN} and {self._AGE_MAX}")

        return age

    def _validate_sibsp(self, sibsp: int) -> int:
        if sibsp < self._SIBSP_MIN or sibsp > self._SIBSP_MAX:
            raise ValueError(f"sibsp must be between {self._SIBSP_MIN} and {self._SIBSP_MAX}")

        return sibsp

    def _validate_parch(self, parch: int) -> int:
        if parch < self._PARCH_MIN or parch > self._PARCH_MAX:
            raise ValueError(f"parch must be between {self._PARCH_MIN} and {self._PARCH_MAX}")

        return parch

    def _validate_fare(self, fare: int) -> int:
        if fare < self._FARE_MIN or fare > self._FARE_MAX:
            raise ValueError(f"sibsp must be between {self._FARE_MIN} and {self._FARE_MAX}")

        return fare

    def get_data(self):
        return [[self._CLASS_VALUE_MAPPER[self.p_class], self.age, self.sibsp, self.parch, self.fare,
                 self._EMBARK_VALUE_MAPPER[self.embark], self._SEX_VALUE_MAPPER[self.sex],
                 self._SEX_VALUE_MAPPER[self.sex]]]




def make_prediction(model_path: str, data_frame: DataFrame) -> dict[str, str]:
    model = pickle.load(open(model_path, 'rb'))
    data = data_frame.get_data()

    survival = model.predict(data)
    confidance = model.predict_proba(data)

    answer = "Czy dana osoba przezyje? {0}".format("Tak" if survival[0] == 1 else "Nie")
    probability = "Pewnosc predykcji {0:.2f}%".format(confidance[0][survival[0]] * 100)

    return {"answer": answer, "probability": probability}


