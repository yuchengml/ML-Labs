# Comparing the mock data
- The variable in the module
- `patch` versus `patch.object`

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