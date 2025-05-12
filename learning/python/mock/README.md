# Comparing the mock data
- The variable in the module
- `patch` versus `patch.object`

# Code
```python
# test_case1.py
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
```

```python
# test_case2.py
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
```

```python
# test_case3.py
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
```

```python
# test_case4.py
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
```

# Experiments
## Execute unit test for each file
```python
# test_case1.py
self.assertEqual(URL, 'http://localhost:5000')
self.assertEqual(result, '`http://localhost:5000` connected')
# ok: 'http://localhost:5000' == 'http://localhost:5000'
# failed: 'connected by mock' != '`http://localhost:5000` connected'

# test_case2.py
self.assertEqual(URL, 'http://localhost:8000')
self.assertEqual(result, 'connected by mock')
# ok: 'http://localhost:8000' == 'http://localhost:8000'
# ok: 'connected by mock' == 'connected by mock'

# test_case3.py
self.assertEqual(URL, 'http://localhost:5000')
self.assertEqual(result, None)
# ok: 'http://localhost:5000' == 'http://localhost:5000'
# ok: None == None

# test_case4.py
self.assertEqual(URL, 'http://localhost:5000')
self.assertEqual(result, 'success')
# ok: 'http://localhost:5000' == 'http://localhost:5000'
# ok: 'success' == 'success'
```

## Execute all unit tests by `python -m unittest discover -s unittest/ -p *.py`
```python
# order: test_case1.py -> test_case2.py -> test_case3.py -> test_case4.py

# test_case1.py
self.assertEqual(URL, 'http://localhost:5000')
self.assertEqual(result, '`http://localhost:5000` connected')
# ok: 'http://localhost:5000' == 'http://localhost:5000'
# failed: 'connected by mock' != '`http://localhost:5000` connected'

# test_case2.py
self.assertEqual(URL, 'http://localhost:8000')
self.assertEqual(result, 'connected by mock')
# failed: 'http://localhost:5000' != 'http://localhost:8000'
# ignore

# test_case3.py
self.assertEqual(URL, 'http://localhost:5000')
self.assertEqual(result, None)
# ok: 'http://localhost:5000' == 'http://localhost:5000'
# ok: None == None

# test_case4.py
self.assertEqual(URL, 'http://localhost:5000')
self.assertEqual(result, 'success')
# ok: 'http://localhost:5000' == 'http://localhost:5000'
# ok: 'success' == 'success'
```