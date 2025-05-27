# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:02:30 2023

@author: merte
"""

class EmptyClass:
  pass

class MyClass():
    x = 3.14
    
ClassObject = MyClass()
print(ClassObject.x)

# we cannot really do a thing with this object. 
# We may want to change x
class MyNewClass():

    # initializer (constructor)
    def __init__(self, x, y=42):
        # two attributes x, y
        self.x = x  # required argument
        self.y = y  # optional argument

my_object = MyNewClass(x=4.0)
print(f'x: {my_object.x}')
print(f'y: {my_object.y}')

# Now we can change attributes, but our class is still a bit useless
class MyDevisionClass():
    
    # initializer (constructor)
    def __init__(self, x, y=42):
        # two attributes x, y
        self.x = x  # required argument
        self.y = y  # optional argument
        
    def print_devision(self):
        x_by_y = self.x / self.y
        print(f'{self.x} / {self.y} = {x_by_y}')
    
my_object = MyDevisionClass(x=4.0) #, y=2.0)
print(f'x: {my_object.x}')
print(f'y: {my_object.y}')
my_object.print_devision()