import unittest
from unittest import mock
from learning.python.mock.client import Client, URL, ApiCaller

import os


class ClientTestCase(unittest.TestCase):

    @mock.patch.dict(os.environ, {'URL': 'http://localhost:8000'})
    @mock.patch.object(Client, 'connect', return_value='connected by mock')
    def test_connect(self, mock_client_connect):

        client = Client()
        result = client.connect()
        self.assertEqual(URL, 'http://localhost:5000')
        self.assertEqual(result, '`http://localhost:5000` connected')  # add assertion here

        api_caller = ApiCaller()
