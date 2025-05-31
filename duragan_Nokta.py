import numpy as np
import math

from ornekFonksiyon2 import f, hessian
from ornekFonksiyon2 import gradient as gradf

#Fonksiyon girilecek, gradient 0 yapan noktalar bulunup hessian hesaplanacak,
#Hessian'ın özdeğerlerin göre yerel minimum yerel maksimum yorumu yapacağız