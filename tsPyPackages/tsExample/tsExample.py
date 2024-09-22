#+------------------------------------------------------------------+
#|                                                 neuropredict2.py |
#|                                                    tony shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "tony shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# import standard python packages
#+-------------------------------------------------------------------
"""
Java C++ type approach in Python

You may be wondering why I didn't call defaultX and defaultY in the object's__init__ method. 
The reason is that for our case I want to assume that the someDefaultComputation methods return
values that vary over time, say a timestamp, and whenever x (or y) is not set (where, for the 
purpose of this example, "not set" means "set to None") I want the value of x's (or y's) default computation.

"""

class Example(object):
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def getX(self):
        return self.x or self.defaultX()

    def getY(self):
        return self.y or self.defaultY()

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y

    def defaultX(self):
        return someDefaultComputationForX()

    def defaultY(self):
        return someDefaultComputationForY()


"""
Re-written with properties ala python. The Java approach is lame in python.
"""

class Example(object):
    def __init__(self, x=None, y=None):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self.x or self.defaultX()

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        return self.y or self.defaultY()

    @y.setter
    def y(self, value):
        self._y = value

    # default{XY} as before.







"""
Class Getter Setter Model1 

Using property() function to achieve getters and setters behaviour
In Python property()is a built-in function that creates and returns a property object. 
A property object has three methods, getter(), setter(), and delete(). 
property() function in Python has four arguments property(fget, fset, fdel, doc), 
fget is a function for retrieving an attribute value. 
fset is a function for setting an attribute value. 
fdel is a function for deleting an attribute value. doc creates a docstring for attribute. 
A property object has three methods, getter(), setter(), and delete() to specify fget, fset
and fdel individually.
"""


# Python program showing a 
# use of property() function  
class Geeks: 
    def __init__(self): 
        self._age = 0
    
    # function to get value of _age 
    def get_age(self): 
        print("getter method called") 
        return self._age 
    
    # function to set value of _age 
    def set_age(self, a): 
        print("setter method called") 
        self._age = a 
    # function to delete _age attribute 
    def del_age(self): 
        del self._age 
    
    age = property(get_age, set_age, del_age)  

#mark = Geeks() 
#mark.age = 10
#print(mark.age) 


"""
Class Getter Setter Model1 


Using @property decorators to achieve getters and setters behaviour

In previous method we used property() function in order to achieve getters and setters behaviour. 
However as mentioned earlier in this post getters and setters are also used for validating the getting
and setting of attributes value. There is one more way to implement property function i.e. by using decorator. 
Python @property is one of the built-in decorators. The main purpose of any decorator is to change your class methods 
or attributes in such a way so that the user of your class no need to make any change in their code. 

"""
# Python program showing the use of 
# @property 

class Geeks: 
    def __init__(self): 
        self._age = 0
    
    # using property decorator 
    # a getter function 
    @property
    def age(self): 
        print("getter method called") 
        return self._age 
    
    # a setter function 
    @age.setter 
    def age(self, a): 
        if(a < 18): 
            raise ValueError("Sorry you age is below eligibility criteria") 
        print("setter method called") 
        self._age = a 

#mark = Geeks() 
#mark.age = 19
#print(mark.age) 

import sys

print("path: ",sys.path )
print("path: ",sys.getprofile )