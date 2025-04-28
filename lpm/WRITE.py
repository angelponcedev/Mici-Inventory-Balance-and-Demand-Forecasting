# search.py (assuming mpl returns the result object)
import numpy as np
import math
import pandas 
import pandas as pd
import sys
from openpyxl import load_workbook

def write(result, producto, cuarto_inicio,output_path):
# --- Call the function and capture the result ---

    #output_path = r"D:/Universidad/TalentLand/LPM/Hackaton DB Final 04.21 W.xlsx"
    wafers = result  # Asegúrate de que esta variable tenga los datos correctos
    
    # --- CARGAR ARCHIVO ---
    book = load_workbook(output_path, data_only=True)
    sheet = book["Wafer Plan"]

    # --- DEFINIR FILA POR PRODUCTO ---
    if producto == "21A":
        start_row = 4
    elif producto == "22B":
        start_row = 5
    elif producto == "23C":
        start_row = 6
    else:
        raise ValueError(f"Producto no reconocido: {producto}")

    headers = [cell.value for cell in sheet[1]]  # Cambié la fila a 1 (fija que es la fila 2 en Excel)

    try:
        col_inicio = headers.index(cuarto_inicio) + 1  # Excel es 1-indexado
        #print(f"\nSe encontró '{cuarto_inicio}' en la columna {col_inicio}.")

        print("\nEscribiendo valores:")
        for i, value in enumerate(wafers):
            col_actual = col_inicio + i
            sheet.cell(row=start_row, column=col_actual, value=value)
            print(f"Celda ({start_row}, {col_actual}) ← {value}")

    except ValueError:
        print(f"\nERROR: No se encontró el cuarto '{cuarto_inicio}' en los encabezados.")
        print("Revisa si el texto coincide exactamente con lo que hay en la fila 1 del Excel (mayúsculas, espacios, etc).")
        
    book.save(output_path)
    print("\n💾 ¡Archivo guardado correctamente!")
    return

def modify(yieldedSupply, tpib, ibesst, producto, cuarto, output_path):
    book = load_workbook(output_path, data_only=True)  # Preserva fórmulas
    sheet = book["Supply_Demand"]
    
    # Buscar filas dinámicamente
    product_rows = {}
    for row in sheet.iter_rows(min_row=2, max_row=19):
        product_id = row[0].value
        attribute = row[1].value
        if product_id == producto and attribute:
            if "Yielded Supply" in attribute:
                product_rows["yield"] = row[0].row
            elif "Total Projected Inventory Balance" in attribute:
                product_rows["tpib"] = row[0].row
            elif "Inventory Balance in excess of SST" in attribute:
                product_rows["ibesst"] = row[0].row
    
    # Validar que se encontraron las filas
    if not product_rows:
        raise KeyError(f"No se encontraron filas para el producto {producto}")
    
    # Buscar columna del trimestre (headers normalizados)
    headers = [str(cell.value).strip().upper() if cell.value else "" for cell in sheet[1]]
    try:
        cuarto_buscar = str(cuarto).strip().upper()
        col_inicio = headers.index(cuarto_buscar) + 1
    except ValueError:
        print(f"Trimestre '{cuarto}' no encontrado. Encabezados disponibles: {headers}")
        return
    
    # Escribir valores (solo en celdas sin fórmulas)
    sheet.cell(row=product_rows["yield"], column=col_inicio, value=float(yieldedSupply))
    sheet.cell(row=product_rows["tpib"], column=col_inicio, value=float(tpib))
    sheet.cell(row=product_rows["ibesst"], column=col_inicio, value=float(ibesst))
    
    book.save(output_path)