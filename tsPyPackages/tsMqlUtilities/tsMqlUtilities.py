#!/usr/bin/env python3  # Uncomment for Linux
# -*- coding: utf-8 -*-  # Uncomment for Linux
"""
Filename: tsMqlMLProcess.py
Description: Load and add files and data parameters.
Author: Tony Shepherd - Xercescloud
Date: 2025-01-24
Version: 1.1
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

    def run_mql_print(self, df: pd.DataFrame, hrows: int = 5, colwidth: int = 20,
                      tablefmt: str = "pretty", floatfmt: str = ".5f",
                      numalign: str = "left", stralign: str = "left") -> None:
        """
        Prints a formatted table of the first few rows of a DataFrame.

        :param df: DataFrame to print
        :param hrows: Number of rows to display
        :param colwidth: Maximum column width
        :param tablefmt: Table format for tabulate
        :param floatfmt: Floating point format
        :param numalign: Numeric alignment
        :param stralign: String alignment
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

       
        print(tabulate(
            df_head,
            showindex=False,
            headers=df_head.columns,
            tablefmt=tablefmt,
            numalign=numalign,
            stralign=stralign,
            floatfmt=floatfmt
        ))
       

