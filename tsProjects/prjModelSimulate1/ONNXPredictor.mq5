//+------------------------------------------------------------------+
//|                                             ONNXPredictor.mq5    |
//|               Sends last N closes to Python and reads output     |
//+------------------------------------------------------------------+
#property script_show_inputs

input string CloseFilePath = "MQL5Files/last_closes.csv";
input string PredictionFilePath = "MQL5Files/mql5_prediction_output.csv";
input string PythonExePath = "python";
input string PythonScript = "LitmusTest1_batch_report_with_mql5_input.py";
input string ModelPath = "your_model.onnx";
input int WindowSize = 5;

void OnStart()
  {
   double closes[];
   ArraySetAsSeries(closes, true);
   CopyClose(Symbol(), PERIOD_M1, 0, WindowSize, closes);
   if(ArraySize(closes) < WindowSize)
     {
      Print("Not enough close data.");
      return;
     }

   // Write closes to CSV
   int file = FileOpen(CloseFilePath, FILE_WRITE|FILE_ANSI|FILE_COMMON, ',');
   if(file != INVALID_HANDLE)
     {
      for(int i = WindowSize - 1; i >= 0; i--)
         FileWrite(file, DoubleToString(closes[i], _Digits));
      FileClose(file);
      Print("✅ Saved last closes to: ", CloseFilePath);
     }
   else
     {
      Print("❌ Failed to write closes.");
      return;
     }

   // Build Python call
   string command = PythonExePath + " " + PythonScript +
                    " --model " + ModelPath +
                    " --closes " + CloseFilePath +
                    " --out " + PredictionFilePath;

   int result = ShellExecuteW(NULL, "open", command, NULL, NULL, SW_HIDE);
   if(result <= 32)
     {
      Print("❌ Failed to run Python script.");
      return;
     }

   Sleep(2000); // Wait for script to complete

   // Read prediction
   file = FileOpen(PredictionFilePath, FILE_READ|FILE_CSV|FILE_COMMON, ',');
   if(file != INVALID_HANDLE)
     {
      string header = FileReadString(file);
      double pred2h = FileReadNumber(file);
      double pred8h = FileReadNumber(file);
      double pred24h = FileReadNumber(file);
      FileClose(file);
      PrintFormat("✅ Prediction:
2h = %.5f
8h = %.5f
24h = %.5f", pred2h, pred8h, pred24h);
     }
   else
      Print("❌ Prediction file not found.");
  }