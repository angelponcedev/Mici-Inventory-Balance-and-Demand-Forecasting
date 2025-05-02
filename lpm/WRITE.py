import numpy as np
import pandas as pd
from openpyxl import load_workbook
import csv
import os

def write(result, product, startQuarter, weeks, outputPath):
    wafers = result
    book = load_workbook(outputPath, data_only=True)
    sheet = book["Wafer Plan"]

    if product == "21A":
        startRow = 4
    elif product == "22B":
        startRow = 5
    elif product == "23C":
        startRow = 6
    else:
        raise ValueError(f"Unrecognized product: {product}")

    headers = [cell.value for cell in sheet[1]]

    try:
        startCol = headers.index(startQuarter) + 1
        for i, value in enumerate(wafers):
            currentCol = startCol + i
            sheet.cell(row=startRow, column=currentCol, value=value)
    except ValueError:
        print(f"ERROR: Quarter '{startQuarter}' not found in headers.")
        return

    book.save(outputPath)
    write2(result, product, startQuarter, weeks)


def modify(yieldedSupply, tpib, ibesst, product, quarter, outputPath):
    book = load_workbook(outputPath, data_only=True)
    sheet = book["Supply_Demand"]

    product_rows = {}
    for row in sheet.iter_rows(min_row=2, max_row=19):
        pid, attr = row[0].value, row[1].value
        if pid == product and attr:
            if "Yielded Supply" in attr:
                product_rows["yield"] = row[0].row
            elif "Total Projected Inventory Balance" in attr:
                product_rows["tpib"] = row[0].row
            elif "Inventory Balance in excess of SST" in attr:
                product_rows["ibesst"] = row[0].row

    if not product_rows:
        raise KeyError(f"No rows found for product {product}")

    headers = [str(cell.value).strip().upper() if cell.value else "" for cell in sheet[1]]
    try:
        quarterKey = str(quarter).strip().upper()
        colStart = headers.index(quarterKey) + 1
    except ValueError:
        print(f"Quarter '{quarter}' not found. Available: {headers}")
        return

    sheet.cell(row=product_rows["yield"], column=colStart, value=float(yieldedSupply))
    sheet.cell(row=product_rows["tpib"], column=colStart, value=float(tpib))
    sheet.cell(row=product_rows["ibesst"], column=colStart, value=float(ibesst))

    book.save(outputPath)

def write2(result, product, startQuarter, weeks):
    basePath = os.path.dirname(__file__)
    filePath = os.path.abspath(os.path.join(basePath, "..", "dataset", "waferPlan.csv"))
    headers = ['Quarter', 'Week', 'Product', 'Wafers']

    wafers = result
    newRows = [[startQuarter, weeks[idx], product, val]
               for idx, val in enumerate(wafers)]

    if os.path.exists(filePath):
        with open(filePath, 'r', newline='', encoding='utf-8') as csvFile:
            reader = csv.reader(csvFile)
            existingRows = list(reader)

        readHeader, body = existingRows[0], existingRows[1:]
        filteredBody = [row for row in body
                        if not (row[0] == startQuarter and row[2] == product)]
    else:
        readHeader, filteredBody = headers, []

    updatedBody = filteredBody + newRows
    with open(filePath, 'w', newline='', encoding='utf-8') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(readHeader)
        writer.writerows(updatedBody)

    print(f"Data for '{product}' in '{startQuarter}' updated.")