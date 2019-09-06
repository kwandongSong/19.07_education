def addition(a, b):
    return a + b

def subtraction(a, b):
    return a - b

print("outside if statement", __name__)
if __name__ == '__main__':
    a = 3
    b = 4
    print(addition(a, b))
    print(subtraction(a, b))
    print("inside if statement", __name__)
