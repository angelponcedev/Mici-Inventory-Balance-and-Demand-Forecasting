import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
import math 


def mpl(n,safetyStockTarget, totalDemand,totalProjectedInventoryBalance, densityPerWafer, yieldPerProduct, availableCapacity, minWeeklyProduction,maxDecrease,weeklyDemandRatio, min, max):
    
    safetyStockTarget = float(safetyStockTarget)
    totalDemand = float(totalDemand)
    totalProjectedInventoryBalance = float(totalProjectedInventoryBalance)
    #totalProjectedInventoryBalance = -totalProjectedInventoryBalance
    densityPerWafer = int(densityPerWafer)  # también para asegurar que sea int nativo
    print(safetyStockTarget)
    print(totalProjectedInventoryBalance)

    integrality = np.ones(n) # RESTRICCION MULTIPLO

    # --- Cálculos Derivados ---
    bytesPerWeek = [densityPerWafer * y for y in yieldPerProduct]
    weeklyDemand = [totalDemand * r for r in weeklyDemandRatio]

    # --- Definición del Problema MILP ---

    # 1. Función Objetivo: Maximizar producción total en bytes
    c = np.array([- 5 * b for b in bytesPerWeek])

    a = -(totalProjectedInventoryBalance - safetyStockTarget - totalDemand)
    b = (totalProjectedInventoryBalance - totalDemand)

    # 2. Variables y Cotas (Bounds)
    # x_t >= minWeeklyProduction
    # x_t <= availableCapacity[t]
    # Las variables x_t representan la cantidad de wafers a producir en la semana t
    lower_bounds_y = [math.ceil(minWeeklyProduction / 5)] * n
    upper_bounds_y = [math.floor(cap / 5) for cap in availableCapacity]
    bounds = Bounds(lb=lower_bounds_y, ub=upper_bounds_y)

    # 3. Indicador de Enteros
    # Queremos que la producción de wafers (x_t) sea entera
    integrality = np.ones(n) # 1 para cada variable x_t

    # 4. Restricciones Lineales (Ax <= b)
    A_ub_list = []
    b_ub_list = []

    # Restricción 1: Inventario Final >= Safety Stock Target
    row_inv_lower = [ -b for b in bytesPerWeek]
    rhs_inv_lower = -(min + safetyStockTarget - totalProjectedInventoryBalance + totalDemand)/5
    A_ub_list.append(row_inv_lower)
    b_ub_list.append(rhs_inv_lower)

    # Límite superior: Inventory Final - SST <= 140M
    # es equivalente a: bytesPerWeek * x_t <= (140M + SST - totalProjectedInventoryBalance + totalDemand)
    row_inv_upper = [ b for b in bytesPerWeek] 
    rhs_inv_upper = (max + safetyStockTarget - totalProjectedInventoryBalance + totalDemand)/5
    A_ub_list.append(row_inv_upper)
    b_ub_list.append(rhs_inv_upper)


    # Restricción 2: Suavización de Producción (Disminución máxima)
    # x_t - x_{t+1} <= maxDecrease  para t = 0 a n-2
    maxDecrease_y = math.floor(maxDecrease / 5) # Nuevo RHS
    for i in range(n - 1):
        # El formato de scipy es A*y <= b.
        # Para y_t - y_{t+1} <= floor(maxDecrease / 5):
        row_smooth = [0] * n
        row_smooth[i] = 1    # Coeficiente para y_t
        row_smooth[i+1] = -1 # Coeficiente para y_{t+1}
        A_ub_list.append(row_smooth)
        b_ub_list.append(maxDecrease_y) # Usar el RHS ajustado

    # Crear objeto LinearConstraint
    if A_ub_list: # Solo si hay restricciones además de las cotas
        constraints = LinearConstraint(A=A_ub_list, ub=b_ub_list)
    else:
        constraints = [] # O None, dependiendo de la versión/manejo de milp

    res = milp(c=c,
            constraints=constraints,
            integrality=integrality,
            bounds=bounds)

    # --- Capturar valores de x y z incluso si no hay éxito ---
    optimal_y = res.x if hasattr(res, 'x') else None
    optimal_z = res.fun + a if res.success else None  # Ajuste para z

    wafers_produced_list = []
    if optimal_y is not None:
        for i in range(n):
            wafers_produced = round(optimal_y[i]) * 5
            wafers_produced_list.append(wafers_produced)

    # Retornar valores (incluso si no hay solución)
    if res.success:
        yieledSuply = sum(avail * bytes for avail, bytes in zip(wafers_produced_list, bytesPerWeek))
        print(f"Valor óptimo (z): {optimal_z}")
        return wafers_produced_list, yieledSuply, yieledSuply + b, -(optimal_z), True
    else:
        # Cálculo de optimal_z como la suma de availableCapacity[i] * bytesPerWeek[i]
        yieledSuply = sum(avail * bytes for avail, bytes in zip(availableCapacity, bytesPerWeek))
        print("No se encontró una solución óptima. Valores aproximados:")
        print(f"Intento de solución (x): {availableCapacity}")
        print(f"Valor de z aproximado: {yieledSuply+a}")
        
        return availableCapacity, yieledSuply, yieledSuply + b, -(yieledSuply + a), False
