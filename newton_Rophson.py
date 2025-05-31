import sympy as sp

def newton_raphson(func_expr, x0, tol=1e-6, max_iter=100):
    x = sp.Symbol('x')
    f = x**2
    f_prime = f.diff(x)  # Türev
    
    # Değerlenecek fonksiyonlar
    f_func = sp.lambdify(x, f, 'math')
    f_prime_func = sp.lambdify(x, f_prime, 'math')
    
    xn = 2
    for i in range(max_iter):
        f_val = f_func(xn)
        f_prime_val = f_prime_func(xn)
        
        if f_prime_val == 0:
            print("Türev sıfır, yöntem durdu.")
            return None
        
        xn_next = xn - f_val / f_prime_val
        
        if abs(xn_next - xn) < tol:
            return xn_next  # Yaklaşık kök
        
        xn = xn_next
    
    print("Maksimum iterasyon sayısına ulaşıldı.")
    return xn
