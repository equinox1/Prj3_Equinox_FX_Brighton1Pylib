#+------------------------------------------------------------------+
#|                                                    tsMqlmod1.pyw
#|                                                    Tony Shepherd |
#|                                                  www.equinox.com |
#+------------------------------------------------------------------+
#property copyright "Tony Shepherd"
#property link      "www.equinox.com"
#property version   "1.01"
#+-------------------------------------------------------------------
# Classes for MQL
#+-------------------------------------------------------------------
#+--------------------------------------------------
# Class CMqlml
#+--------------------------------------------------
class CMqlml:
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

    def defaultX(self):
        return someDefaultComputationForX()

    def defaultY(self):
        return someDefaultComputationForY()