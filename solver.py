import sympy as sp
import re

def solve_expression(expression):
    try:
        # Clean the expression
        expression = expression.strip()
        expression = expression.replace('x', '*')
        expression = expression.replace('X', '*')

        # Check if it's an equation (contains =)
        if '=' in expression:
            parts = expression.split('=')
            left = parts[0].strip()
            right = parts[1].strip()
            x = sp.Symbol('x')
            equation = sp.Eq(sp.sympify(left), sp.sympify(right))
            solution = sp.solve(equation, x)
            return f"x = {solution}"
        else:
            # Simple arithmetic
            result = sp.sympify(expression)
            return str(sp.simplify(result))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    # Test it
    print(solve_expression("2+3"))       
    print(solve_expression("10/2"))       
    print(solve_expression("2**3"))       
    print(solve_expression("2x+3=7"))     
