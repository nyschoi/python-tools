# ex
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
[x for x in lst if x > 4 if x % 2 == 0]

lst = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
[x ** 2 for x in lst if x % 3 == 0 or x % 5 == 0]

lst = [['a', 'b', 'c'], [1, 2, 3]]
[item * 2 for line in lst for item in line]


# list comprehension with nested for loop
lst = [1, 2, 3]
lst_rev = [3, 2, 1]
# [(1, 3), (1, 2), (1, 1), (2, 3), (2, 2), (2, 1), (3, 3), (3, 2), (3, 1)]
[(k, v) for k in lst for v in lst_rev]

# map 대용으로
animals = ['tiger', 'Lion', 'doG', 'CAT']
list(map(str.upper, animals))

[str.upper(x) for x in animals]

# list가 아닌 set comprehension
lst = [-3, -2, -1, 1, 2, 3, 4, 5]
{x ** 2 for x in lst}

# list가 아닌 dict comprehension
lst = [-3, -2, -1, 1, 2, 3, 4, 5]
{k: k ** 2 for k in lst}

# tuple comprehension은 없고 generator comprehension은 있다. 더 좋다!
numbers = list(range(10_000_000))
squares = [x ** x for x in numbers]
squares.__sizeof__()
squares_gen = (x ** x for x in numbers)
squares_gen.__sizeof__()
sum(squares_gen) == sum(squares)


# 참고: Ternary Operator in Python
# https://www.geeksforgeeks.org/ternary-operator-in-python/
# [on_true] if [expression] else [on_false]
a, b = 10, 20
a if a < b else b

# Ternary op.를 list comprehension에 응용할 수 있다.
numbers = [1, 2, 3, 4, 5]
[pow(x, 2) if x % 2 else 'HEHE' for x in numbers]
# store all values of lst in list e whose values are greater than 4 – else if the values are less than 4 then it will store the string “less than 4” in its place.
[x if x > 4 else 'less then 4' for x in lst]

# store the string “Two” if the value is divisible by 2. Or if the value is divisible by 3 we are storing “Three”, else we are storing “not 2 & 3”.
# ['Two' if x %2==0 else x%3==0 'Three' for x in lst]
['Two' if x % 2 == 0 else 'Three' if x % 3 == 0 else x for x in lst]
