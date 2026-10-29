from typing import List, Callable
import numpy as np

Vector = List[float]

def rosenbroke(x: Vector): return (1-x[0])**2 + 100*(x[1] - x[0]**2)**2

def eckley(x: Vector): return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x[0]**2 + x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0]) + np.cos(2*pi*x[1]))) + exp(1) + 20

def himmelbau(x: Vector): return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2

def rostrigin(x: Vector): return 20 + (x[0]**2 - 10*np.cos(2*np.pi*x[0])) + (x[1]**2 - 10*np.cos(2*np.pi*x[1]))

def izom(x:Vector): return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))