#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqlMLProcess.py
Description: Load and add files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1 (Revised)
"""

import os
import sys
import logging
import textwrap
import numpy as np
import pandas as pd
import tensorflow as tf
from tabulate import tabulate

# Set up logger if not already configured
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class CUtilities:
    def __init__(self, **kwargs):
        """Initialize data processing class."""
        self._logger = logging.getLogger(__name__)
        self._logger.info("Initializing Utilities class...")
        self.df: pd.DataFrame = kwargs.get("df", None)
        self.hrows: int = kwargs.get("hrows", 5)
        self.colwidth: int = kwargs.get("colwidth", 20)
        self.tablefmt: str = kwargs.get("tablefmt", "pretty")
        self.floatfmt: str = kwargs.get("floatfmt", ".5f")
        self.numalign: str = kwargs.get("numalign", "left")
        self.stralign: str = kwargs.get("stralign", "left")

    def run_mql_print(self, df: pd.DataFrame, df_name: str = "DataFrame", hrows: int = 5, colwidth: int = 20,
                      app=None,
                      tablefmt: str = "pretty", floatfmt: str = ".5f",
                      numalign: str = "left", stralign: str = "left") -> None:
        """
        Prints a formatted table of the first few rows of a DataFrame.

        :param df: DataFrame to print
        :param df_name: Name of the DataFrame for logging
        :param hrows: Number of rows to display
        :param colwidth: Maximum column width for text wrapping
        :param tablefmt: Table format for tabulate
        :param floatfmt: Floating point format for numbers
        :param numalign: Numeric alignment in the table
        :param stralign: String alignment in the table
        """
        if df is None or df.empty:
            self._logger.warning("DataFrame is empty or None.")
            print("No data available to display.")
            return

        print(f"Print Table: first {min(hrows, len(df))} rows (total rows: {len(df)})")

        def wrap_column_data(column, width):
            return column.apply(lambda x: '\n'.join(textwrap.wrap(str(x), width)))

        df_head = df.head(hrows).copy()
        df_head = df_head.apply(lambda col: wrap_column_data(col, colwidth))

        
        formatted_table = tabulate(
        df_head,
        showindex=False,
        headers=df_head.columns,
        tablefmt=tablefmt,
        numalign=numalign,
        stralign=stralign,
        floatfmt=floatfmt
         )
        self._logger.info(f"Programme App finish: {app} Data from file: {df_name}, shape: {df.shape}\n" + formatted_table)

        


# Example usage if this file is run as a script
if __name__ == "__main__":
    # Create sample DataFrame for demonstration
    data = {
        "Column1": [f"Row{i} data" for i in range(1, 11)],
        "Column2": np.random.rand(10),
        "Column3": [f"Some longer text that might need wrapping {i}" for i in range(1, 11)]
    }
    sample_df = pd.DataFrame(data)

    # Instantiate the utility class and print the sample DataFrame
    util = CUtilities(df=sample_df)
    util.run_mql_print(sample_df, df_name="Sample DataFrame", hrows=5, colwidth=30)
