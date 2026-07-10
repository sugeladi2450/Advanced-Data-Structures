from typing import Optional, Dict


class ExampleKeyValueStore:
    def __init__(self, data: Dict[int, str]) -> None:
        self._data = data

    def get(self, key: int) -> Optional[str]:
        return self._data.get(key)


example_kv_store = ExampleKeyValueStore(
    {
        1: "apple",
        3: "banana",
        5: "cherry",
        7: "date",
        10: "elderberry",
    }
)
