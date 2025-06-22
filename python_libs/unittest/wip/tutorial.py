def divide(x, y):
    if y == 0:
        raise ValueError('չի կարելի, չէ, չի կարելի')
    res = x / y
    return res

def is_prime(x):
    if x == 8:
        return "Panir"
    if x < 2:
        return False
    for i in range(2, x):
        if x % i == 0:
            return False
    return True
    

# import random
# def generate_random_num():
#     return random.randint(0, 100)
