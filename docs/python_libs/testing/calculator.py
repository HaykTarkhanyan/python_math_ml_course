def add(a: int, b: int) -> int:
    return a + b 

def div(a: int, b: int) -> float:
    if b == 0:
        print("AAAAA")
        # raise ValueError("b must be non-zero")
    return a / b

# test_calculator # test

# ------------- Coverage -------------
# # pytest --cov=calculator # Stmt = # executable lines
def subtract(a: int, b: int) -> int:
    return a - b

def a():
    pass
# -------------- Debugging ------------
# def is_prime(n: int) -> bool:
#     if n <= 1:
#         return False
#     for i in range(2, int(n**0.5) + 1):
#         if n % i == 0:
#             return False
#     return True


# done = []
# for a in range(-1, 10):
#     if a > 0:
#         a = int(a)
#         print(f"Is {a} prime? {is_prime(a)}")
#         done.append(a)
#     else:
#         print("Enter a positive integer greater than 0.")

# print("All numbers entered:", done)