import unittest
from unittest import mock
from learning.python.mock.client import ApiCaller
import os


class ApiCallerTestCase(unittest.TestCase):

    @mock.patch.dict(os.environ, {'URL': 'http://localhost:5000'})
    @mock.patch.object(ApiCaller, 'call', return_value=None)
    def test_api_call(self, mock_api_caller_call):
        from learning.python.mock.client import URL
        caller = ApiCaller()
        result = caller.call()
        self.assertEqual(URL, 'http://localhost:5000')
        self.assertEqual(result, None)  # add assertion here
