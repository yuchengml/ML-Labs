from typing import Union, List, Dict

class DataStruct:
    def __init__(self):
        pass

class Trackable:
    def __init__(self, parent: 'MetricsLogger', data: Union[List, Dict, DataStruct]):
        self.parent = parent
        self.data = data

    def update(self):
        """Appends this attribute's data to the parent's collection."""
        self.parent.collection.append(self.data)


class MetricsLogger:
    def __init__(self):
        self.collection = []
        self.a = Trackable(self, list())
        self.b = Trackable(self, dict())
        self.c = Trackable(self, DataStruct())


if __name__ == '__main__':
    logger = MetricsLogger()
    logger.a.update()  # Appends logger.a to collection
    logger.a.update()  # Appends logger.a again to collection
    logger.b.update()  # Appends logger.b to collection
    logger.c.update()  # Appends logger.b to collection
    logger.c.update()  # Appends logger.b to collection

    print(logger.collection)  # Output: [[], [], {}]
