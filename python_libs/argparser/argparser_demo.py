import argparse

def divide(x, y, details=False):
    if details:
        print(f"Run divide with {x} and {y}")
    
    return x / y

# argparse 
a = argparse.ArgumentParser(description="Divide two numbers")  
a.add_argument("x", type=int, help="First number") 
a.add_argument("y", type=int, help="Second number")
a.add_argument("--details", action="store_true", help="Print details")

args = a.parse_args()

print(divide(args.x, args.y, args.details))