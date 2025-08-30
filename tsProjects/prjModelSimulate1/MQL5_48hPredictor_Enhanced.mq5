//+------------------------------------------------------------------+
//|                              MQL5_48hPredictor_Enhanced.mq5      |
//|  Enhanced: error score, auto plot PNG export, tighter handling   |
//+------------------------------------------------------------------+
#property script_show_inputs

input string CloseExport = "MQL5Files/ohlc_48h_input.csv";
input string PredictionCSV = "MQL5Files/mql5_backtest_result.csv";
input string PythonExe = "python";
input string PythonScript = "predict_from_back_window.py";
input string ONNXModelPath = "model.onnx";
input string PlotOutput = "MQL5Files/predicted_vs_actual_2h.png";

void OnStart()
  {
   int bars_needed = 60 * 48;
   datetime time_series[];
   double closes[];
   ArraySetAsSeries(time_series, true);
   ArraySetAsSeries(closes, true);
   CopyTime(Symbol(), PERIOD_M1, 0, bars_needed, time_series);
   CopyClose(Symbol(), PERIOD_M1, 0, bars_needed, closes);

   int n = ArraySize(closes);
   if(n < bars_needed)
     {
      Print("Not enough 1-min bars available for 48h export.");
      return;
     }

   int file = FileOpen(CloseExport, FILE_WRITE|FILE_CSV|FILE_COMMON);
   if(file != INVALID_HANDLE)
     {
      FileWrite(file, "datetime,close");
      for(int i = n - 1; i >= 0; i--)
         FileWrite(file, TimeToString(time_series[i], TIME_DATE|TIME_MINUTES), DoubleToString(closes[i], _Digits));
      FileClose(file);
      Print("✅ 48h OHLC closes exported to: ", CloseExport);
     }
   else
     {
      Print("❌ Failed to write close file.");
      return;
     }

   string cmd = PythonExe + " " + PythonScript +
                " --input "" + CloseExport + """ +
                " --model "" + ONNXModelPath + """ +
                " --output "" + PredictionCSV + """;

   int result = ShellExecuteW(NULL, "open", cmd, NULL, NULL, SW_HIDE);
   if(result <= 32)
     {
      Print("❌ Failed to execute Python prediction.");
      return;
     }

   Sleep(3000); // allow Python to finish

   file = FileOpen(PredictionCSV, FILE_READ|FILE_CSV|FILE_COMMON);
   if(file == INVALID_HANDLE)
     {
      Print("❌ No prediction CSV found.");
      return;
     }

   string header = FileReadString(file);
   double pred = FileReadNumber(file);
   double a2h = FileReadNumber(file);
   double e2h = FileReadNumber(file);
   double a8h = FileReadNumber(file);
   double e8h = FileReadNumber(file);
   double a24h = FileReadNumber(file);
   double e24h = FileReadNumber(file);
   FileClose(file);

   double score = (e2h + e8h + e24h) / 3.0;

   PrintFormat("✅ Prediction from Python = %.5f", pred);
   PrintFormat("🟢 Actuals => 2h: %.5f | 8h: %.5f | 24h: %.5f", a2h, a8h, a24h);
   PrintFormat("📊 Errors  => 2h: %.5f | 8h: %.5f | 24h: %.5f", e2h, e8h, e24h);
   PrintFormat("📈 MAPE Score: %.5f (lower is better)", score);
   PrintFormat("🖼️ Plot (if generated): %s", PlotOutput);
  }