# 자주 쓰는 것들
## 1. List Comprehension Using If-Else
```python
my_list = [
    'Multiple of 6' if i % 6 == 0
    else 'Multiple of 2' if i % 2 == 0
    else 'Multiple of 3' if i % 3 == 0
    else i for i in range(1, 20)
]
print(my_list)
```
## 2. Merging Two Dictionaries
```python
my_dict1 = {'a': 1, 'b': 2, 'c': 3}
my_dict2 = {'d': 4, 'e': 5, 'f': 6}

# Method 1
result = {**my_dict1, **my_dict2}
print(result)

# Method 2
result = my_dict1.copy()
result.update(my_dict2)
print(result)

# Method 3
result = {key: value for d in (my_dict1, my_dict2) for key, value in d.items()}
print(result)
```

## 3. File Handling
```python
# Open a file
f = open('filename.txt')
# Read from a file
f = open('filename.txt', 'r')
# To read the whole file
print(f.read())
# To read single line
print(f.readline())
# Write to a file
f = open('filename.txt', 'w')
f.write('Writing into a file \n')
# Closing a file
f.close()
```

## 4. Calculating Execution Time
```python
import time
start_time = time.time()
# printing all even numbers till 20
for i in range(20):
    if i % 2 == 0:
        print(i, end=" ")
end_time = time.time()
time_taken = end_time - start_time
print("\nTime: ", time_taken)
```

## 5. Sorting a List of Dictionaries
```python
person = [
    {'name': 'alice', 'age': 22, 'id': 92345},
    {'name': 'bob', 'age': 24, 'id': 52353},
    {'name': 'tom', 'age': 23, 'id': 62257},
]

# Method 1
person.sort(key=lambda item: item.get("id"))  #  return type is None as the changes are in place
print(person)

# Method 2
person = sorted(person, key=lambda item: item.get("id"))
print(person)
```

## 6. Finding Highest Frequency Element
```python
my_list = [8, 4, 8, 2, 2, 5, 8, 0, 3, 5, 2, 5, 8, 9, 3, 8]
print("Most frequent item:", max(set(my_list), key=my_list.count))
```

## 7. Error Handling
```python
num1, num2 = 2, 0
try:
    print(num1 / num2)
except ZeroDivisionError:
    print("Exception! Division by Zero not permitted.")
else:
    print('no exception raised')
finally:
    print("Finally block.")
```
## 8. Finding Substring in List of Strings
```python
records = [
    "Vani Gupta, University of Hyderabad",
    "Elon Musk, Tesla",
    "Bill Gates, Microsoft",
    "Steve Jobs, Apple",
]

# Method 1
name = "Vani"
for record in records:
    if record.find(name) >= 0:
        print(record)

# Method 2
name = "Musk"
for record in records:
    if name in record:
        print(record)
```

## 9. String Formatting
```python
language = "Python"

# Method 1
print(language + " is my favourite programming language.")

# Method 2
print(f"I code in {language}")

# Method 3
print("%s is very easy to learn." % (language))

# Method 4
print("I like the {} programming language.".format(language))
```
## 10. Flattening a List
```python
# method 1
ugly_list = [10, 12, 36, [41, 59, 63], [77], 81, 93]
flat = []
for i in ugly_list:
    if isinstance(i, list):
        flat.extend(i)
    else:
        flat.append(i)
print(flat)

# method 2
from iteration_utilities import deepflatten

# if you only have one depth nested_list, use this
def flatten(l):
  return [item for sublist in l for item in sublist]

l = [[1,2,3],[3]]
print(flatten(l))
# [1, 2, 3, 3]

# if you don't know how deep the list is nested
l = [[1,2,3],[4,[5],[6,7]],[8,[9,[10]]]]

print(list(deepflatten(l, depth=3)))
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
```

## 11. Reversing a String
```python
# Reversing a string using slicing
my_string = "ABCDE"
reversed_string = my_string[::-1]
print(reversed_string)
# Output
# EDCBA
```

## 12. Using rhe Title Case (First Letter Caps)
```python
my_string = "my name is chaitanya baweja"

# using the title() function of string class
new_string = my_string.title()

print(new_string)

# Output
# My Name Is Chaitanya Baweja
```
## 13. Finding Unique Elements in a String
```python
my_string = "aavvccccddddeee"
# converting the string to a set
temp_set = set(my_string)
# stitching set into a string using join
new_string = ''.join(temp_set)
print(new_string)
```

## 14. Printing a String or a List n Times
```python
n = 3 # number of repetitions
my_string = "abcd"
my_list = [1,2,3]
print(my_string*n)
# abcdabcdabcd
print(my_list*n)
# [1,2,3,1,2,3,1,2,3]
```

## 15. Swap Values Between Two Variables
```python
a = 1
b = 2

a, b = b, a

print(a) # 2
print(b) # 1
```

## 16. Split a String Into a List of Substrings
```python
string_1 = "My name is Chaitanya Baweja"
string_2 = "sample/ string 2"

# default separator ' '
print(string_1.split())
# ['My', 'name', 'is', 'Chaitanya', 'Baweja']

# defining separator as '/'
print(string_2.split('/'))
# ['sample', ' string 2']
```
## 17. Combining a List of Strings Into a Single String
```python
list_of_strings = ['My', 'name', 'is', 'Chaitanya', 'Baweja']

# Using join with the comma separator
print(','.join(list_of_strings))

# Output
# My,name,is,Chaitanya,Baweja
```

## 18. Check If a Given String Is a Palindrome or Not
```python
my_string = "abcba"

if my_string == my_string[::-1]:
    print("palindrome")
else:
    print("not palindrome")
```

## 19. Frequency of Elements in a List
```python
# finding frequency of each element in a list
from collections import Counter

my_list = ['a','a','b','b','b','c','d','d','d','d','d']
count = Counter(my_list) # defining a counter object

print(count) # Of all elements
# Counter({'d': 5, 'b': 3, 'a': 2, 'c': 1})

print(count['b']) # of individual element
# 3

print(count.most_common(1)) # most frequent element
# [('d', 5)]
```

## 20. Find Whether Two Strings are Anagrams
```python
from collections import Counter

str_1, str_2, str_3 = "acbde", "abced", "abcda"
cnt_1, cnt_2, cnt_3  = Counter(str_1), Counter(str_2), Counter(str_3)

if cnt_1 == cnt_2:
    print('1 and 2 anagram')
if cnt_1 == cnt_3:
    print('1 and 3 anagram')
```

## 21. Using Enumerate to Get Index/Value Pairs
```python
my_list = ['a', 'b', 'c', 'd', 'e']

for index, value in enumerate(my_list):
    print('{0}: {1}'.format(index, value))

# 0: a
# 1: b
# 2: c
# 3: d
# 4: e
```
## 22. Check the Memory Usage of an Object

```python
import sys

num = 21

print(sys.getsizeof(num))

# In Python 2, 24
# In Python 3, 28
```
## 23. Sampling From a List
```python
# method 1
import random
my_list = ['a', 'b', 'c', 'd', 'e']
num_samples = 2
samples = random.sample(my_list,num_samples)
print(samples)
# [ 'a', 'e'] this will have any 2 random values

# method 2: recommended the secrets library for generating random samples for cryptography purposes
import secrets                              # imports secure module.
secure_random = secrets.SystemRandom()      # creates a secure random object.

my_list = ['a','b','c','d','e']
num_samples = 2

samples = secure_random.sample(my_list, num_samples)

print(samples)
# [ 'e', 'd'] this will have any 2 random values

```

## 24. Digitize
```python
num = 123456

# using map
list_of_digits = list(map(int, str(num)))

print(list_of_digits)
# [1, 2, 3, 4, 5, 6]

# using list comprehension
list_of_digits = [int(x) for x in str(num)]

print(list_of_digits)
# [1, 2, 3, 4, 5, 6]
```

## 25. Check for Uniqueness
```python
def unique(l):
    if len(l)==len(set(l)):
        print("All elements are unique")
    else:
        print("List has duplicates")

unique([1,2,3,4])
# All elements are unique

unique([1,1,2,3])
# List has duplicates
```

# list/dict/set comprehension
## list comprehension
```python
lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
[x for x in lst if x > 4 if x % 2 == 0]

lst = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]
[x ** 2 for x in lst if x % 3 == 0 or x % 5 == 0]

lst = [['a', 'b', 'c'], [1, 2, 3]]
[item * 2 for line in lst for item in line]

* list comprehension with nested for loop
lst = [1, 2, 3]
lst_rev = [3, 2, 1]
# [(1, 3), (1, 2), (1, 1), (2, 3), (2, 2), (2, 1), (3, 3), (3, 2), (3, 1)]
[(k, v) for k in lst for v in lst_rev]

* map 대용으로
animals = ['tiger', 'Lion', 'doG', 'CAT']
list(map(str.upper, animals))

[str.upper(x) for x in animals]

## set comprehension
lst = [-3, -2, -1, 1, 2, 3, 4, 5]
{x ** 2 for x in lst}

## list가 아닌 dict comprehension
lst = [-3, -2, -1, 1, 2, 3, 4, 5]
{k: k ** 2 for k in lst}

## tuple comprehension은 없고 generator comprehension은 있다. 더 좋다!
numbers = list(range(10_000_000))
squares = [x ** x for x in numbers]
squares.__sizeof__()
squares_gen = (x ** x for x in numbers)
squares_gen.__sizeof__()
sum(squares_gen) == sum(squares)
```

## 참고: Ternary Operator in Python

```python
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
```

# Todo
## https://wikidocs.net/book/536

## https://github.com/vinta/awesome-python#awesome-python/
- algorithm...
- 