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

mark = Geeks() 
mark.age = 10
print(mark.age) 


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

mark = Geeks() 
mark.age = 19
print(mark.age) 