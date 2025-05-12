import unittest
from unittest import mock
# from learning.python.mock.client import Client, URL

import os


class ClientTestCase(unittest.TestCase):

    @mock.patch.dict(os.environ, {'URL': 'http://localhost:8000'})
    @mock.patch('learning.python.mock.client.Client.connect', return_value='connected by mock')
    def test_connect(self, mock_client_connect):
        from learning.python.mock.client import Client, URL
        client = Client()
        result = client.connect()
        self.assertEqual(URL, 'http://localhost:8000')
        self.assertEqual(result, 'connected by mock')  # add assertion here
