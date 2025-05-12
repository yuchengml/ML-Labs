import unittest
from unittest import mock


class Foo:
    def foo(self):
        pass


# with mock.patch.object(Foo, 'foo') as mock_foo:
with mock.patch.object(Foo, 'foo', autospec=True) as mock_foo:
    mock_foo.return_value = 'foo'
    foo = Foo()
    print(foo.foo())

# mock_foo.assert_called_once_with(foo)
# class MyTestCase(unittest.TestCase):
#     def test_something(self):
#         self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    # unittest.main()
    mock_foo.assert_called_once_with(foo)
