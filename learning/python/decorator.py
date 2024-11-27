import time  # For the sleep functionality


def delay_execution(seconds):
    """Decorator to delay the execution of a function by a given number of seconds."""
    def decorator(func):
        def wrap(*args, **kwargs):
            print(f"Delaying execution of '{func.__name__}' for {seconds} second(s)...")
            time.sleep(seconds)
            result = func(*args, **kwargs)
            return result
        return wrap
    return decorator


@delay_execution(seconds=3)  # Delay the execution by 3 seconds
def say_hello(name):
    print(f"Hello, {name}!")


@delay_execution(seconds=1)  # Delay the execution by 1 second
def add_numbers(a, b):
    print(f"The result of {a} + {b} is {a + b}")


if __name__ == "__main__":
    say_hello("Alice")  # Waits for 3 seconds before printing the message
    # Output:
    # Delaying execution of 'say_hello' for 3 second(s)...
    # Hello, Alice!

    add_numbers(5, 7)  # Waits for 1 second before printing the result
    # Output:
    # Delaying execution of 'add_numbers' for 1 second(s)...
    # The result of 5 + 7 is 12
