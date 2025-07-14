from modules.CodeBlockExecutor import trace
# Example Python script with output markers

# Basic arithmetic
x = 10
y = 20
trace(x + y)
#> 30

# String operations
name = "World"
greeting = f"Hello, {name}!"
trace(greeting)
#> Hello, World!

# List operations
numbers = [1, 2, 3, 4, 5]
squared = [n**2 for n in numbers]
trace(squared)
#> [1, 4, 9, 16, 25]

# Function definition and call
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Calculate first 10 fibonacci numbers
fib_sequence = [fibonacci(i) for i in range(10)]
trace(fib_sequence)
#> [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Print statements

# Error handling example (this will show an error)
# trace(undefined_variable)


# Dictionary operations
person = {"name": "Alice", "age": 30, "city": "New York"}
trace(person["age"])
#> 30

# More complex example with imports
import math
trace(math.pi * 2)
#> 6.283185307179586
