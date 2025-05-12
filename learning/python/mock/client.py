import os

URL = os.environ.get("URL", 'http://localhost:5000')


class Client:
    def __init__(self):
        self.url = URL

    def connect(self):
        return f"`{self.url}` connected"

class ApiCaller:
    def __init__(self):
        self.url = URL

    def call(self):
        raise NotImplementedError
