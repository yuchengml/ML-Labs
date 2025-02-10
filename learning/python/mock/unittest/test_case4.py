import unittest
from unittest import mock

import os


class ApiCallerTestCase(unittest.TestCase):

    @mock.patch.dict(os.environ, {'URL': 'http://localhost:5000'})
    @mock.patch('learning.python.mock.client.ApiCaller.call', return_value='success')
    def test_api_call(self, mock_api_caller_call):
        from learning.python.mock.client import URL, ApiCaller
        # caller = ApiCaller()
        # result = caller.call()
        result = mock_api_caller_call()
        self.assertEqual(URL, 'http://localhost:5000')
        self.assertEqual(result, 'success')  # add assertion here
