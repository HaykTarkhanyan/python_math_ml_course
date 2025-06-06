# import helper # 1
# from helper import square, cube, password # 2
# from helper import * # 3
# import helper as h # 4
# from helper import square as s # 5
from files_helper import square

tiv = int(input("Մուտքագրեք թիվը: "))

print(tiv)
print(square(tiv))



# print(helper.square(tiv)) # 1
# print(square(tiv)) # 2
# print(cube(tiv))
# print(password)

# print(h.cube(tiv)) # 4
