# from scipy.optimize import fsolve
# import math
#
# def equations(p):
#     x, y = p
#     return (x+y**2-4, math.exp(x) + x*y - 3)
#
# x, y =  fsolve(equations, (1, 1))
#
# print ((x, y), equations((x, y)))

from ortools.linear_solver import pywraplp
solver = pywraplp.Solver.CreateSolver('GLOP')
p = solver.NumVar(0, 1, 'p')
q = solver.NumVar(0, 1, 'q')

solver.Add(5*q+10*(1-q)-8*q-3*(1-q) == 0)
solver.Add(5*p+10*(1-p)-7*p-2*(1-p) == 0)

solver.Maximize(p+q)

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', solver.Objective().Value())
    print('q =', p.solution_value())
    print('p =', q.solution_value())
else:
    print('The problem does not have an optimal solution.')

print('\nAdvanced usage:')
print('Problem solved in %f milliseconds' % solver.wall_time())
print('Problem solved in %d iterations' % solver.iterations())
