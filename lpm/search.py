import pandas as pd
import os
from typing import Dict, Any, List, Tuple
from lpm.lpm import mpl
from lpm.WRITE import write , modify
from lpm.lpm import mpl0

def load_excel_data(file_path: str) -> Dict[str, pd.DataFrame]:
    """Carga todas las hojas necesarias del Excel en un dict."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Archivo no encontrado: {file_path}")

    excel_data: Dict[str, pd.DataFrame] = {}

    # 1) Supply & Demand
    excel_data["supply_demand"] = pd.read_excel(
        file_path, sheet_name="Supply_Demand", engine="openpyxl"
    )

    # 2) Density per Wafer
    excel_data["density"] = pd.read_excel(
        file_path, sheet_name="Density per Wafer", engine="openpyxl"
    )

    # 3) Weekly Demand Ratio
    excel_data["weekly_ratio_df"] = pd.read_excel(
        file_path, sheet_name="Weekly Demand Ratio", engine="openpyxl"
    )

    # 4) Yield (headers en dos filas: nivel0=trimestre, nivel1=fecha)
    yield_df = pd.read_excel(
        file_path, sheet_name="Yield", header=[0, 1], engine="openpyxl"
    )
    # Renombrar las dos primeras columnas a Meta/Product ID y Meta/Attribute
    cols: List[Tuple[Any, Any]] = list(yield_df.columns)
    if len(cols) >= 2:
        cols[0] = ("Meta", "Product ID")
        cols[1] = ("Meta", "Attribute")
        yield_df.columns = pd.MultiIndex.from_tuples(cols)
    excel_data["yield_df"] = yield_df

    # 5) Boundary Conditions (igual: nivel0=trimestre, nivel1=semana)
    boundary_df = pd.read_excel(
        file_path,
        sheet_name="Boundary Conditions",
        header=[0, 1],
        engine="openpyxl",
    )
    cols_b: List[Tuple[Any, Any]] = list(boundary_df.columns)
    if len(cols_b) >= 2:
        cols_b[0] = ("Meta", "Product ID")
        cols_b[1] = ("Meta", "Attribute")
        boundary_df.columns = pd.MultiIndex.from_tuples(cols_b)
    excel_data["boundary_df"] = boundary_df
    return excel_data

def get_weeks_from_columns(df: pd.DataFrame, quarter_label: str) -> List[str]:
    """Extrae los nombres de las semanas (WW_XX) del nivel1 en las columnas del trimestre indicado."""
    return [
        col[1] for col in df.columns
        if isinstance(col, tuple) and col[0] == quarter_label and str(col[1]).startswith("WW_")
    ]


def get_product_data(
    data: Dict[str, pd.DataFrame],
    product_id: str,
    nquarter: int,
    year: int
) -> Dict[str, Any]:
    """Extrae los datos para `product_id` en Q{nquarter} {year}, pero toma
       inventory_balance de Q{nquarter-1} (con ajuste de año si nquarter=1)."""
    product_data: Dict[str, Any] = {}

    # Construimos las etiquetas "Qx YY" para el actual y el anterior
    def make_label(nq: int, yr: int) -> str:
        # Handle years 100+ by displaying only last two digits (e.g., 100 -> "00")
        if yr >= 100:
            yr_display = yr % 100  # Get the last two digits
            fmt = f"{yr_display:02d}"  # Always display as 2 digits with leading zero if needed
        else:
            fmt = f"{yr:02d}"  # Display as 2 digits
        
        return f"Q{nq} {fmt}"

    curr_label = make_label(nquarter, year)
    # calcula trimestre anterior
    if nquarter == 1:
        prev_q, prev_year = 4, year - 1
    else:
        prev_q, prev_year = nquarter - 1, year
    prev_label = make_label(prev_q, prev_year)
    print(prev_label)
    print(curr_label)
    # — Supply & Demand —
    df = data["supply_demand"]

    def _get_attr(attr_name: str, label: str) -> Any:
        sel = df[(df["Product ID"] == product_id) & (df["Attribute"] == attr_name)]
        if sel.empty:
            raise KeyError(f"⚠️ {attr_name} no encontrado para {product_id}")
        if label not in sel.columns:
            raise KeyError(f"⚠️ Columna '{label}' no existe en Supply_Demand")
        return sel.iloc[0][label]

    # safety_stock y total_demand del trimestre actual
    product_data["safety_stock"] = _get_attr("Safety Stock Target", curr_label)
    product_data["total_demand"]  = _get_attr("EffectiveDemand", curr_label)
    # inventory_balance del trimestre anterior
    product_data["inventory_balance"] = _get_attr(
        "Total Projected Inventory Balance",
        prev_label
    )

    # — Density per Wafer —
    df_den = data["density"]
    if product_id not in df_den.columns:
        raise KeyError(f"⚠️ Producto {product_id} no existe en Density per Wafer")
    product_data["density"] = df_den.loc[0, product_id]

    # — Weekly Demand Ratio —
    product_data["weekly_ratio"] = data["weekly_ratio_df"].iloc[0].tolist()

    # — Yield — 
    ydf = data["yield_df"]
    row = ydf[ydf[("Meta", "Product ID")] == product_id]
    if row.empty:
        raise KeyError(f"⚠️ {product_id} no encontrado en Yield")
    # filtramos columnas cuyo nivel0 coincide con la etiqueta actual
    cols = [col for col in ydf.columns if col[0] == curr_label]
    if not cols:
        raise KeyError(f"⚠️ No se encontraron columnas para '{curr_label}' en Yield")
    cols_sorted = sorted(cols, key=lambda x: pd.to_datetime(x[1]))
    product_data["yield_values"] = row.loc[:, cols_sorted].values.flatten().tolist()

    # — Available Capacity (Boundary Conditions) —
    bdf = data["boundary_df"]
    rowb = bdf[
        (bdf[("Meta", "Product ID")] == product_id)
        & (bdf[("Meta", "Attribute")] == "Available Capacity")
    ]
    if rowb.empty:
        raise KeyError(
            f"⚠️ Available Capacity no encontrada para {product_id}"
        )
    cap_cols = [col for col in bdf.columns if col[0] == curr_label]
    if not cap_cols:
        raise KeyError(
            f"⚠️ No se encontraron columnas para '{curr_label}' en Boundary Conditions"
        )
    # orden por número de semana WW_##   
    def _week_num(code: str) -> int:
        return int(str(code).replace("WW_", ""))
    cap_cols_sorted = sorted(cap_cols, key=lambda x: _week_num(x[1]))
    product_data["available_capacity"] = rowb.loc[:, cap_cols_sorted].values.flatten().tolist()
    product_data["week_labels"] = [col[1] for col in cap_cols_sorted]

    return product_data

# ----------------------------
# Bloque principal (ejemplo)
# ----------------------------
def search(FILE_PATH,years,final_quarter):

    #FILE_PATH = "D:/Universidad/TalentLand/LPM/Hackaton DB Final 04.21.xlsx"
    #FILE_PATH = "../dataset/Hackaton DB Final 04.21.xlsx"
    PRODUCTS = ["21A","22B","23C"]  # ,"22B","23C"
    START_YEAR = 95     # para '95

    all_data = load_excel_data(FILE_PATH)

    for i in range(10):  # 8 years
        current_year = START_YEAR + i
        
        for j in range(4):  # 4 quarters per year
            current_quarter = j + 1
            
            # Skip Q1 and Q2 of the first year (95)
            if current_year == START_YEAR and current_quarter < 3:
                continue
            if current_year == (95+years) and current_quarter >final_quarter:
                break
                
            # Format quarter label
            if current_year > 99:
                # Handle years 100+
                display_year = f"0{current_year - 100}" if current_year < 110 else f"{current_year - 100}"
                #current_year=display_year
                QUARTER = f"Q{current_quarter} {display_year}"
            else:
                QUARTER = f"Q{current_quarter} {current_year}"
            
            PRODUCT = PRODUCTS[0]
            
            # Get product data for the current quarter
            try:
                all_data = load_excel_data(FILE_PATH)
                pdict = get_product_data(all_data, PRODUCT, current_quarter, current_year)
                print(f"Processing: {PRODUCT}, {QUARTER}")
                
                WP, YS, TPIB, IBESST,S = mpl(
                    n=len(pdict["available_capacity"]),
                    safetyStockTarget=pdict["safety_stock"],
                    totalDemand=pdict["total_demand"],
                    totalProjectedInventoryBalance=pdict["inventory_balance"],
                    densityPerWafer=pdict["density"],
                    yieldPerProduct=pdict["yield_values"],
                    availableCapacity=pdict["available_capacity"],
                    minWeeklyProduction=350,
                    maxDecrease=560,
                    weeklyDemandRatio=pdict["weekly_ratio"],
                    min=70000000,
                    max=140000000,
                )
                
                print(f"Results: YS={YS}, TPIB={TPIB}, IBESST={IBESST}, for {PRODUCT}, {QUARTER}")
                
                # Write results to files
                print(pdict)

                write(WP, PRODUCT, QUARTER, pdict["week_labels"], FILE_PATH)
                modify(YS, TPIB, IBESST, PRODUCT, QUARTER, FILE_PATH)
                #print(pdict)
                
            except KeyError as e:
                print(f"Error processing {PRODUCT}, {QUARTER}: {e}")
                continue  # Skip to next iteration if data is missing

    START_YEAR = 95
        # — Procesar productos 2 y 3 con lógica de rescate de capacidad —
    for l in range(1, 3):
        PRODUCT = PRODUCTS[l]
        for i in range(10):  # 8 años
            current_year = START_YEAR + i
            for j in range(4):  # 4 trimestres
                current_quarter = j + 1
                # Saltar Q1 y Q2 del primer año
                if current_year == START_YEAR and current_quarter < 3:
                    continue
                if current_year == (95+years) and current_quarter >final_quarter:
                    break

                # Formatear etiqueta de trimestre
                if current_year > 99:
                    display_year = (
                        f"0{current_year - 100}"
                        if current_year < 110
                        else f"{current_year - 100}"
                    )
                else:
                    display_year = f"{current_year}"
                QUARTER = f"Q{current_quarter} {display_year}"

                try:
                    all_data = load_excel_data(FILE_PATH)
                    pdict = get_product_data(all_data, PRODUCT, current_quarter, current_year)
                    print(f"Processing: {PRODUCT}, {QUARTER}")

                    # 1) Primera llamada a mpl
                    WP, YS, TPIB, IBESST, S = mpl(
                        n=len(pdict["available_capacity"]),
                        safetyStockTarget=pdict["safety_stock"],
                        totalDemand=pdict["total_demand"],
                        totalProjectedInventoryBalance=pdict["inventory_balance"],
                        densityPerWafer=pdict["density"],
                        yieldPerProduct=pdict["yield_values"],
                        availableCapacity=pdict["available_capacity"],
                        minWeeklyProduction=350,
                        maxDecrease=560,
                        weeklyDemandRatio=pdict["weekly_ratio"],
                        min=70000000,
                        max=140000000,
                    )

                    # 2) Si falla (S == False) y hay un producto anterior, aplico rescate
                    if not S and l > 0:
                        prev_prod = PRODUCTS[l - 1]
                        pdict_prev = get_product_data(
                            all_data, prev_prod, current_quarter, current_year
                        )
                        # mpl del anterior para obtener WP_prev
                        WP_prev, _, _, _, S_prev = mpl(
                            n=len(pdict_prev["available_capacity"]),
                            safetyStockTarget=pdict_prev["safety_stock"],
                            totalDemand=pdict_prev["total_demand"],
                            totalProjectedInventoryBalance=pdict_prev["inventory_balance"],
                            densityPerWafer=pdict_prev["density"],
                            yieldPerProduct=pdict_prev["yield_values"],
                            availableCapacity=pdict_prev["available_capacity"],
                            minWeeklyProduction=350,
                            maxDecrease=560,
                            weeklyDemandRatio=pdict_prev["weekly_ratio"],
                            min=70000000,
                            max=140000000,
                        )

                        # Calcular sobrantes semana a semana
                        leftover = [
                            prev_cap - used
                            for prev_cap, used in zip(
                                pdict_prev["available_capacity"], WP_prev
                            )
                        ]

                        # Ajustar capacidad disponible del producto actual
                        pdict["available_capacity"] = [
                            curr_cap + extra
                            for curr_cap, extra in zip(
                                pdict["available_capacity"], leftover
                            )
                        ]

                        # 3) Reintentar mpl con la capacidad ajustada
                        WP, YS, TPIB, IBESST, S = mpl(
                            n=len(pdict["available_capacity"]),
                            safetyStockTarget=pdict["safety_stock"],
                            totalDemand=pdict["total_demand"],
                            totalProjectedInventoryBalance=pdict["inventory_balance"],
                            densityPerWafer=pdict["density"],
                            yieldPerProduct=pdict["yield_values"],
                            availableCapacity=pdict["available_capacity"],
                            minWeeklyProduction=350,
                            maxDecrease=560,
                            weeklyDemandRatio=pdict["weekly_ratio"],
                            min=70000000,
                            max=140000000,
                        )

                    # 4) Escritura de resultados
                    print(f"Results: YS={YS}, TPIB={TPIB}, IBESST={IBESST}, S={S}")

                    write(WP, PRODUCT, QUARTER, pdict["week_labels"], FILE_PATH)
                    modify(YS, TPIB, IBESST, PRODUCT, QUARTER, FILE_PATH)

                except KeyError as e:
                    print(f"Error processing {PRODUCT}, {QUARTER}: {e}")
                    continue

def search_combined(FILE_PATH,years,final_quarter):

    #FILE_PATH = "D:/Universidad/TalentLand/LPM/Hackaton DB Final 04.21.xlsx"
    #FILE_PATH = "../dataset/Hackaton DB Final 04.21.xlsx"
    PRODUCTS = ["21A","22B","23C"]  # ,"22B","23C"
    START_YEAR = 95     # para '95
    quarter_counter = 0

    all_data = load_excel_data(FILE_PATH)

    for i in range(years):  # 8 years
        current_year = START_YEAR + i
        
        for j in range(4):  # 4 quarters per year
            current_quarter = j + 1
            
            # Skip Q1 and Q2 of the first year (95)
            if current_year == START_YEAR and current_quarter < 3:
                continue
            if current_year == (95+years) and current_quarter >final_quarter:
                break

            quarter_counter += 1
                
            # Format quarter label
            if current_year > 99:
                # Handle years 100+
                display_year = f"0{current_year - 100}" if current_year < 110 else f"{current_year - 100}"
                #current_year=display_year
                QUARTER = f"Q{current_quarter} {display_year}"
            else:
                QUARTER = f"Q{current_quarter} {current_year}"
            
            PRODUCT = PRODUCTS[0]
            
            if quarter_counter <=34:
            # Get product data for the current quarter
                try:
                    all_data = load_excel_data(FILE_PATH)
                    pdict = get_product_data(all_data, PRODUCT, current_quarter, current_year)
                    print(f"Processing: {PRODUCT}, {QUARTER}")
                    
                    WP, YS, TPIB, IBESST,S = mpl(
                        n=len(pdict["available_capacity"]),
                        safetyStockTarget=pdict["safety_stock"],
                        totalDemand=pdict["total_demand"],
                        totalProjectedInventoryBalance=pdict["inventory_balance"],
                        densityPerWafer=pdict["density"],
                        yieldPerProduct=pdict["yield_values"],
                        availableCapacity=pdict["available_capacity"],
                        minWeeklyProduction=350,
                        maxDecrease=560,
                        weeklyDemandRatio=pdict["weekly_ratio"],
                        min=70000000,
                        max=140000000,
                    )
                    
                    print(f"Results: YS={YS}, TPIB={TPIB}, IBESST={IBESST}, for {PRODUCT}, {QUARTER}")
                    
                    # Write results to files
                    print(pdict)

                    write(WP, PRODUCT, QUARTER,  pdict["week_labels"], FILE_PATH)
                    modify(YS, TPIB, IBESST, PRODUCT, QUARTER, FILE_PATH)
                    #print(pdict)
                    
                except KeyError as e:
                    print(f"Error processing {PRODUCT}, {QUARTER}: {e}")
                    continue  # Skip to next iteration if data is missing

            else:
                try:
                    all_data = load_excel_data(FILE_PATH)
                    pdict = get_product_data(all_data, PRODUCT, current_quarter, current_year)
                    print(f"Processing: {PRODUCT}, {QUARTER}")
                    
                    WP, YS, TPIB, IBESST,S = mpl0(
                        n=len(pdict["available_capacity"]),
                        safetyStockTarget=pdict["safety_stock"],
                        totalDemand=pdict["total_demand"],
                        totalProjectedInventoryBalance=pdict["inventory_balance"],
                        densityPerWafer=pdict["density"],
                        yieldPerProduct=pdict["yield_values"],
                        availableCapacity=pdict["available_capacity"],
                        minWeeklyProduction=350,
                        maxDecrease=560,
                        weeklyDemandRatio=pdict["weekly_ratio"],
                    )
                    
                    print(f"Results: YS={YS}, TPIB={TPIB}, IBESST={IBESST}, for {PRODUCT}, {QUARTER}")
                    
                    # Write results to files
                    print(pdict)

                    write(WP, PRODUCT, QUARTER, pdict["week_labels"], FILE_PATH)
                    modify(YS, TPIB, IBESST, PRODUCT, QUARTER, FILE_PATH)
                    #print(pdict)
                    
                except KeyError as e:
                    print(f"Error processing {PRODUCT}, {QUARTER}: {e}")
                    continue  # Skip to next iteration if data is missing

    START_YEAR = 95
        # — Procesar productos 2 y 3 con lógica de rescate de capacidad —
    for l in range(1, 3):
        PRODUCT = PRODUCTS[l]
        for i in range(10):  # 8 años
            current_year = START_YEAR + i
            for j in range(4):  # 4 trimestres
                current_quarter = j + 1
                # Saltar Q1 y Q2 del primer año
                if current_year == START_YEAR and current_quarter < 3:
                    continue
                if current_year == (95+years) and current_quarter >final_quarter:
                    break

                quarter_counter += 1

                # Formatear etiqueta de trimestre
                if current_year > 99:
                    display_year = (
                        f"0{current_year - 100}"
                        if current_year < 110
                        else f"{current_year - 100}"
                    )
                else:
                    display_year = f"{current_year}"
                QUARTER = f"Q{current_quarter} {display_year}"

                if quarter_counter <=34:
                    try:
                        all_data = load_excel_data(FILE_PATH)
                        pdict = get_product_data(all_data, PRODUCT, current_quarter, current_year)
                        print(f"Processing: {PRODUCT}, {QUARTER}")

                        # 1) Primera llamada a mpl
                        WP, YS, TPIB, IBESST, S = mpl(
                            n=len(pdict["available_capacity"]),
                            safetyStockTarget=pdict["safety_stock"],
                            totalDemand=pdict["total_demand"],
                            totalProjectedInventoryBalance=pdict["inventory_balance"],
                            densityPerWafer=pdict["density"],
                            yieldPerProduct=pdict["yield_values"],
                            availableCapacity=pdict["available_capacity"],
                            minWeeklyProduction=350,
                            maxDecrease=560,
                            weeklyDemandRatio=pdict["weekly_ratio"],
                            min=70000000,
                            max=140000000,
                        )

                        # 2) Si falla (S == False) y hay un producto anterior, aplico rescate
                        if not S and l > 0:
                            prev_prod = PRODUCTS[l - 1]
                            pdict_prev = get_product_data(
                                all_data, prev_prod, current_quarter, current_year
                            )
                            # mpl del anterior para obtener WP_prev
                            WP_prev, _, _, _, S_prev = mpl(
                                n=len(pdict_prev["available_capacity"]),
                                safetyStockTarget=pdict_prev["safety_stock"],
                                totalDemand=pdict_prev["total_demand"],
                                totalProjectedInventoryBalance=pdict_prev["inventory_balance"],
                                densityPerWafer=pdict_prev["density"],
                                yieldPerProduct=pdict_prev["yield_values"],
                                availableCapacity=pdict_prev["available_capacity"],
                                minWeeklyProduction=350,
                                maxDecrease=560,
                                weeklyDemandRatio=pdict_prev["weekly_ratio"],
                                min=70000000,
                                max=140000000,
                            )

                            # Calcular sobrantes semana a semana
                            leftover = [
                                prev_cap - used
                                for prev_cap, used in zip(
                                    pdict_prev["available_capacity"], WP_prev
                                )
                            ]

                            # Ajustar capacidad disponible del producto actual
                            pdict["available_capacity"] = [
                                curr_cap + extra
                                for curr_cap, extra in zip(
                                    pdict["available_capacity"], leftover
                                )
                            ]

                            # 3) Reintentar mpl con la capacidad ajustada
                            WP, YS, TPIB, IBESST, S = mpl(
                                n=len(pdict["available_capacity"]),
                                safetyStockTarget=pdict["safety_stock"],
                                totalDemand=pdict["total_demand"],
                                totalProjectedInventoryBalance=pdict["inventory_balance"],
                                densityPerWafer=pdict["density"],
                                yieldPerProduct=pdict["yield_values"],
                                availableCapacity=pdict["available_capacity"],
                                minWeeklyProduction=350,
                                maxDecrease=560,
                                weeklyDemandRatio=pdict["weekly_ratio"],
                                min=70000000,
                                max=140000000,
                            )

                        # 4) Escritura de resultados
                        print(f"Results: YS={YS}, TPIB={TPIB}, IBESST={IBESST}, S={S}")

                        write(WP, PRODUCT, QUARTER, pdict["week_labels"], FILE_PATH)
                        modify(YS, TPIB, IBESST, PRODUCT, QUARTER, FILE_PATH)

                    except KeyError as e:
                        print(f"Error processing {PRODUCT}, {QUARTER}: {e}")
                        continue

                else:
                    try:
                        all_data = load_excel_data(FILE_PATH)
                        pdict = get_product_data(all_data, PRODUCT, current_quarter, current_year)
                        print(f"Processing: {PRODUCT}, {QUARTER}")
                        
                        WP, YS, TPIB, IBESST,S = mpl0(
                            n=len(pdict["available_capacity"]),
                            safetyStockTarget=pdict["safety_stock"],
                            totalDemand=pdict["total_demand"],
                            totalProjectedInventoryBalance=pdict["inventory_balance"],
                            densityPerWafer=pdict["density"],
                            yieldPerProduct=pdict["yield_values"],
                            availableCapacity=pdict["available_capacity"],
                            minWeeklyProduction=350,
                            maxDecrease=560,
                            weeklyDemandRatio=pdict["weekly_ratio"],
                        )
                        
                        print(f"Results: YS={YS}, TPIB={TPIB}, IBESST={IBESST}, for {PRODUCT}, {QUARTER}")
                        
                        # Write results to files
                        print(pdict)

                        write(WP, PRODUCT, QUARTER, pdict["week_labels"], FILE_PATH)
                        modify(YS, TPIB, IBESST, PRODUCT, QUARTER, FILE_PATH)
                        #print(pdict)
                        
                    except KeyError as e:
                        print(f"Error processing {PRODUCT}, {QUARTER}: {e}")
                        continue  # Skip to next iteration if data is missing