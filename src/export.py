"""Excel report export for simulation results."""

from io import BytesIO

import numpy as np
import pandas as pd


def _autofit_columns(worksheet) -> None:
    """Set each column width to fit its longest cell."""
    for column_cells in worksheet.columns:
        longest = max(len(str(cell.value)) if cell.value is not None else 0
                      for cell in column_cells)
        worksheet.column_dimensions[column_cells[0].column_letter].width = longest + 3


def build_excel_report(ticker: str, inputs: dict, stats: dict, paths: np.ndarray) -> bytes:
    """
    Build an in-memory Excel workbook with three sheets:
    Summary, Daily_Percentiles and Final_Prices. Returns the file as bytes.
    """
    # Sheet 1: inputs and headline results
    rows = [(label, round(value, 4) if isinstance(value, float) else value)
            for label, value in inputs.items()]
    rows += [
        ("", ""),
        ("Results at final day", ""),
        ("10th percentile (Rs)", round(float(stats["p10"]), 2)),
        ("Median (Rs)", round(float(stats["p50"]), 2)),
        ("90th percentile (Rs)", round(float(stats["p90"]), 2)),
        ("Mean (Rs)", round(float(stats["mean"]), 2)),
    ]
    summary = pd.DataFrame(rows, columns=["Metric", "Value"])

    # Sheet 2: percentile bands for every trading day
    daily = pd.DataFrame({
        "Day": np.arange(paths.shape[0]),
        "10th percentile": np.percentile(paths, 10, axis=1),
        "Median": np.percentile(paths, 50, axis=1),
        "90th percentile": np.percentile(paths, 90, axis=1),
        "Mean": paths.mean(axis=1),
    }).round(2)

    # Sheet 3: final price of every simulation
    final_prices = pd.DataFrame({
        "Simulation": np.arange(1, paths.shape[1] + 1),
        "Final Price (Rs)": paths[-1].round(2),
    })

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        daily.to_excel(writer, sheet_name="Daily_Percentiles", index=False)
        final_prices.to_excel(writer, sheet_name="Final_Prices", index=False)
        for worksheet in writer.sheets.values():
            _autofit_columns(worksheet)

    return buffer.getvalue()