//+------------------------------------------------------------------+
//|                                                  PythonBridge.mq4 |
//|                                 MT4-Python ZeroMQ Bridge EA      |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "QuantProject"
#property link      "https://github.com/quantproject"
#property version   "1.00"
#property strict

// 導入ZeroMQ庫
#import "libzmq.dll"
   int zmq_ctx_new();
   int zmq_ctx_destroy(int context);
   int zmq_socket(int context, int type);
   int zmq_close(int socket);
   int zmq_bind(int socket, string endpoint);
   int zmq_connect(int socket, string endpoint);
   int zmq_send(int socket, string message, int length, int flags);
   int zmq_recv(int socket, string &buffer, int length, int flags);
   int zmq_setsockopt(int socket, int option, int &value, int size);
#import

// ZeroMQ socket types
#define ZMQ_REQ 3
#define ZMQ_REP 4
#define ZMQ_PUB 1
#define ZMQ_SUB 2

// ZeroMQ options
#define ZMQ_RCVTIMEO 27
#define ZMQ_SNDTIMEO 28
#define ZMQ_SUBSCRIBE 6

// 全局變數
int context = 0;
int rep_socket = 0;   // 回應Python請求
int pub_socket = 0;   // 發布數據流

// 配置參數
input string InpPythonHost = "tcp://localhost";  // Python主機地址
input int InpRepPort = 5555;                     // REP端口(接收Python請求)
input int InpPubPort = 5557;                     // PUB端口(發布數據)
input bool InpEnableTickData = true;             // 啟用Tick數據發送
input bool InpEnableOHLC = true;                 // 啟用K線數據發送
input int InpSendInterval = 100;                 // 數據發送間隔(毫秒)

// 狀態變數
bool isConnected = false;
datetime lastTickTime = 0;
datetime lastBarTime = 0;
int tickCount = 0;
string lastError = "";

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("PythonBridge EA 正在初始化...");
   
   // 創建ZeroMQ context
   context = zmq_ctx_new();
   if(context == 0)
   {
      Print("錯誤: 無法創建ZeroMQ context");
      return INIT_FAILED;
   }
   
   // 創建REP socket (回應Python請求)
   rep_socket = zmq_socket(context, ZMQ_REP);
   if(rep_socket == 0)
   {
      Print("錯誤: 無法創建REP socket");
      zmq_ctx_destroy(context);
      return INIT_FAILED;
   }
   
   // 創建PUB socket (發布數據)
   pub_socket = zmq_socket(context, ZMQ_PUB);
   if(pub_socket == 0)
   {
      Print("錯誤: 無法創建PUB socket");
      zmq_close(rep_socket);
      zmq_ctx_destroy(context);
      return INIT_FAILED;
   }
   
   // 設置socket超時
   int timeout = 1000; // 1秒
   zmq_setsockopt(rep_socket, ZMQ_RCVTIMEO, timeout, 4);
   zmq_setsockopt(rep_socket, ZMQ_SNDTIMEO, timeout, 4);
   
   // 綁定sockets
   string rep_endpoint = InpPythonHost + ":" + IntegerToString(InpRepPort);
   string pub_endpoint = InpPythonHost + ":" + IntegerToString(InpPubPort);
   
   if(zmq_bind(rep_socket, rep_endpoint) != 0)
   {
      Print("錯誤: 無法綁定REP socket到 " + rep_endpoint);
      Cleanup();
      return INIT_FAILED;
   }
   
   if(zmq_bind(pub_socket, pub_endpoint) != 0)
   {
      Print("錯誤: 無法綁定PUB socket到 " + pub_endpoint);
      Cleanup();
      return INIT_FAILED;
   }
   
   isConnected = true;
   Print("PythonBridge EA 初始化成功");
   Print("REP socket: " + rep_endpoint);
   Print("PUB socket: " + pub_endpoint);
   
   // 發送初始化消息
   SendStatus("EA_INITIALIZED");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   Print("PythonBridge EA 正在關閉...");
   
   // 發送關閉消息
   SendStatus("EA_CLOSING");
   
   // 清理資源
   Cleanup();
   
   Print("PythonBridge EA 已關閉");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   if(!isConnected) return;
   
   // 處理Python請求
   ProcessPythonRequests();
   
   // 發送Tick數據
   if(InpEnableTickData)
   {
      SendTickData();
   }
   
   // 發送K線數據
   if(InpEnableOHLC && IsNewBar())
   {
      SendOHLCData();
   }
   
   tickCount++;
}

//+------------------------------------------------------------------+
//| 處理Python請求                                                   |
//+------------------------------------------------------------------+
void ProcessPythonRequests()
{
   string buffer = "";
   string response = "";
   
   // 非阻塞接收
   int result = zmq_recv(rep_socket, buffer, 1024, 1); // ZMQ_DONTWAIT = 1
   
   if(result > 0)
   {
      Print("收到Python請求: " + buffer);
      
      // 解析並處理請求
      response = HandleRequest(buffer);
      
      // 發送回應
      zmq_send(rep_socket, response, StringLen(response), 0);
   }
}

//+------------------------------------------------------------------+
//| 處理請求並返回響應                                               |
//+------------------------------------------------------------------+
string HandleRequest(string request)
{
   // 解析JSON請求
   string command = GetJsonValue(request, "command");
   
   if(command == "HEARTBEAT")
   {
      return CreateJsonResponse("ok", "heartbeat");
   }
   else if(command == "GET_ACCOUNT_INFO")
   {
      return GetAccountInfo();
   }
   else if(command == "GET_POSITIONS")
   {
      return GetOpenPositions();
   }
   else if(command == "OPEN_ORDER")
   {
      string symbol = GetJsonValue(request, "symbol");
      int type = (int)StringToInteger(GetJsonValue(request, "type"));
      double lots = StringToDouble(GetJsonValue(request, "lots"));
      double price = StringToDouble(GetJsonValue(request, "price"));
      double sl = StringToDouble(GetJsonValue(request, "sl"));
      double tp = StringToDouble(GetJsonValue(request, "tp"));
      
      return OpenOrder(symbol, type, lots, price, sl, tp);
   }
   else if(command == "CLOSE_ORDER")
   {
      int ticket = (int)StringToInteger(GetJsonValue(request, "ticket"));
      return CloseOrder(ticket);
   }
   else if(command == "MODIFY_ORDER")
   {
      int ticket = (int)StringToInteger(GetJsonValue(request, "ticket"));
      double sl = StringToDouble(GetJsonValue(request, "sl"));
      double tp = StringToDouble(GetJsonValue(request, "tp"));
      
      return ModifyOrder(ticket, sl, tp);
   }
   else if(command == "GET_MARKET_DATA")
   {
      string symbol = GetJsonValue(request, "symbol");
      return GetMarketData(symbol);
   }
   else
   {
      return CreateJsonResponse("error", "Unknown command: " + command);
   }
}

//+------------------------------------------------------------------+
//| 發送Tick數據                                                     |
//+------------------------------------------------------------------+
void SendTickData()
{
   if(lastTickTime == TimeCurrent()) return;
   
   string data = "{";
   data += "\"type\":\"tick\",";
   data += "\"symbol\":\"" + Symbol() + "\",";
   data += "\"time\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\",";
   data += "\"bid\":" + DoubleToString(Bid, Digits) + ",";
   data += "\"ask\":" + DoubleToString(Ask, Digits) + ",";
   data += "\"spread\":" + IntegerToString(MarketInfo(Symbol(), MODE_SPREAD)) + ",";
   data += "\"volume\":" + DoubleToString(Volume[0], 0);
   data += "}";
   
   zmq_send(pub_socket, data, StringLen(data), 0);
   lastTickTime = TimeCurrent();
}

//+------------------------------------------------------------------+
//| 發送K線數據                                                      |
//+------------------------------------------------------------------+
void SendOHLCData()
{
   string data = "{";
   data += "\"type\":\"ohlc\",";
   data += "\"symbol\":\"" + Symbol() + "\",";
   data += "\"period\":\"" + PeriodToString(Period()) + "\",";
   data += "\"time\":\"" + TimeToString(Time[1], TIME_DATE|TIME_SECONDS) + "\",";
   data += "\"open\":" + DoubleToString(Open[1], Digits) + ",";
   data += "\"high\":" + DoubleToString(High[1], Digits) + ",";
   data += "\"low\":" + DoubleToString(Low[1], Digits) + ",";
   data += "\"close\":" + DoubleToString(Close[1], Digits) + ",";
   data += "\"volume\":" + DoubleToString(Volume[1], 0);
   data += "}";
   
   zmq_send(pub_socket, data, StringLen(data), 0);
}

//+------------------------------------------------------------------+
//| 獲取帳戶信息                                                     |
//+------------------------------------------------------------------+
string GetAccountInfo()
{
   string info = "{";
   info += "\"status\":\"ok\",";
   info += "\"data\":{";
   info += "\"account_number\":" + IntegerToString(AccountNumber()) + ",";
   info += "\"balance\":" + DoubleToString(AccountBalance(), 2) + ",";
   info += "\"equity\":" + DoubleToString(AccountEquity(), 2) + ",";
   info += "\"margin\":" + DoubleToString(AccountMargin(), 2) + ",";
   info += "\"free_margin\":" + DoubleToString(AccountFreeMargin(), 2) + ",";
   info += "\"profit\":" + DoubleToString(AccountProfit(), 2) + ",";
   info += "\"leverage\":" + IntegerToString(AccountLeverage()) + ",";
   info += "\"currency\":\"" + AccountCurrency() + "\"";
   info += "}}";
   
   return info;
}

//+------------------------------------------------------------------+
//| 獲取持倉信息                                                     |
//+------------------------------------------------------------------+
string GetOpenPositions()
{
   string positions = "{\"status\":\"ok\",\"data\":[";
   bool first = true;
   
   for(int i = 0; i < OrdersTotal(); i++)
   {
      if(OrderSelect(i, SELECT_BY_POS, MODE_TRADES))
      {
         if(!first) positions += ",";
         
         positions += "{";
         positions += "\"ticket\":" + IntegerToString(OrderTicket()) + ",";
         positions += "\"symbol\":\"" + OrderSymbol() + "\",";
         positions += "\"type\":" + IntegerToString(OrderType()) + ",";
         positions += "\"lots\":" + DoubleToString(OrderLots(), 2) + ",";
         positions += "\"open_price\":" + DoubleToString(OrderOpenPrice(), Digits) + ",";
         positions += "\"sl\":" + DoubleToString(OrderStopLoss(), Digits) + ",";
         positions += "\"tp\":" + DoubleToString(OrderTakeProfit(), Digits) + ",";
         positions += "\"profit\":" + DoubleToString(OrderProfit(), 2) + ",";
         positions += "\"open_time\":\"" + TimeToString(OrderOpenTime(), TIME_DATE|TIME_SECONDS) + "\"";
         positions += "}";
         
         first = false;
      }
   }
   
   positions += "]}";
   return positions;
}

//+------------------------------------------------------------------+
//| 開單函數                                                         |
//+------------------------------------------------------------------+
string OpenOrder(string symbol, int type, double lots, double price, double sl, double tp)
{
   int ticket = -1;
   
   // 設置symbol
   if(symbol != Symbol())
   {
      return CreateJsonResponse("error", "Can only trade current symbol");
   }
   
   // 根據類型開單
   if(type == OP_BUY)
   {
      ticket = OrderSend(symbol, OP_BUY, lots, Ask, 3, sl, tp, "PythonBridge", 0, 0, clrGreen);
   }
   else if(type == OP_SELL)
   {
      ticket = OrderSend(symbol, OP_SELL, lots, Bid, 3, sl, tp, "PythonBridge", 0, 0, clrRed);
   }
   else
   {
      // 掛單交易
      ticket = OrderSend(symbol, type, lots, price, 3, sl, tp, "PythonBridge", 0, 0, clrBlue);
   }
   
   if(ticket > 0)
   {
      string response = "{\"status\":\"ok\",\"ticket\":" + IntegerToString(ticket) + "}";
      return response;
   }
   else
   {
      int error = GetLastError();
      string response = "{\"status\":\"error\",\"message\":\"" + ErrorDescription(error) + "\"}";
      return response;
   }
}

//+------------------------------------------------------------------+
//| 平倉函數                                                         |
//+------------------------------------------------------------------+
string CloseOrder(int ticket)
{
   if(OrderSelect(ticket, SELECT_BY_TICKET))
   {
      double closePrice = (OrderType() == OP_BUY) ? Bid : Ask;
      
      if(OrderClose(ticket, OrderLots(), closePrice, 3, clrYellow))
      {
         return CreateJsonResponse("ok", "Order closed: " + IntegerToString(ticket));
      }
      else
      {
         int error = GetLastError();
         return CreateJsonResponse("error", ErrorDescription(error));
      }
   }
   else
   {
      return CreateJsonResponse("error", "Order not found: " + IntegerToString(ticket));
   }
}

//+------------------------------------------------------------------+
//| 修改訂單                                                         |
//+------------------------------------------------------------------+
string ModifyOrder(int ticket, double sl, double tp)
{
   if(OrderSelect(ticket, SELECT_BY_TICKET))
   {
      if(OrderModify(ticket, OrderOpenPrice(), sl, tp, 0, clrNONE))
      {
         return CreateJsonResponse("ok", "Order modified: " + IntegerToString(ticket));
      }
      else
      {
         int error = GetLastError();
         return CreateJsonResponse("error", ErrorDescription(error));
      }
   }
   else
   {
      return CreateJsonResponse("error", "Order not found: " + IntegerToString(ticket));
   }
}

//+------------------------------------------------------------------+
//| 獲取市場數據                                                     |
//+------------------------------------------------------------------+
string GetMarketData(string symbol)
{
   if(symbol == "") symbol = Symbol();
   
   string data = "{";
   data += "\"status\":\"ok\",";
   data += "\"data\":{";
   data += "\"symbol\":\"" + symbol + "\",";
   data += "\"bid\":" + DoubleToString(MarketInfo(symbol, MODE_BID), Digits) + ",";
   data += "\"ask\":" + DoubleToString(MarketInfo(symbol, MODE_ASK), Digits) + ",";
   data += "\"spread\":" + IntegerToString((int)MarketInfo(symbol, MODE_SPREAD)) + ",";
   data += "\"digits\":" + IntegerToString((int)MarketInfo(symbol, MODE_DIGITS)) + ",";
   data += "\"point\":" + DoubleToString(MarketInfo(symbol, MODE_POINT), Digits);
   data += "}}";
   
   return data;
}

//+------------------------------------------------------------------+
//| 發送狀態消息                                                     |
//+------------------------------------------------------------------+
void SendStatus(string status)
{
   string data = "{";
   data += "\"type\":\"status\",";
   data += "\"status\":\"" + status + "\",";
   data += "\"time\":\"" + TimeToString(TimeCurrent(), TIME_DATE|TIME_SECONDS) + "\"";
   data += "}";
   
   zmq_send(pub_socket, data, StringLen(data), 0);
}

//+------------------------------------------------------------------+
//| 創建JSON響應                                                     |
//+------------------------------------------------------------------+
string CreateJsonResponse(string status, string message)
{
   return "{\"status\":\"" + status + "\",\"message\":\"" + message + "\"}";
}

//+------------------------------------------------------------------+
//| 從JSON字符串獲取值                                               |
//+------------------------------------------------------------------+
string GetJsonValue(string json, string key)
{
   string searchKey = "\"" + key + "\":";
   int start = StringFind(json, searchKey);
   
   if(start == -1) return "";
   
   start += StringLen(searchKey);
   
   // 跳過空格
   while(start < StringLen(json) && StringSubstr(json, start, 1) == " ")
      start++;
   
   // 檢查是否為字符串值
   bool isString = (StringSubstr(json, start, 1) == "\"");
   
   if(isString)
   {
      start++; // 跳過開頭引號
      int end = StringFind(json, "\"", start);
      if(end == -1) return "";
      return StringSubstr(json, start, end - start);
   }
   else
   {
      // 數值或布爾值
      int end = StringFind(json, ",", start);
      if(end == -1) end = StringFind(json, "}", start);
      if(end == -1) return "";
      return StringSubstr(json, start, end - start);
   }
}

//+------------------------------------------------------------------+
//| 檢查是否為新K線                                                  |
//+------------------------------------------------------------------+
bool IsNewBar()
{
   if(lastBarTime != Time[0])
   {
      lastBarTime = Time[0];
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| 將週期轉換為字符串                                               |
//+------------------------------------------------------------------+
string PeriodToString(int period)
{
   switch(period)
   {
      case PERIOD_M1:  return "M1";
      case PERIOD_M5:  return "M5";
      case PERIOD_M15: return "M15";
      case PERIOD_M30: return "M30";
      case PERIOD_H1:  return "H1";
      case PERIOD_H4:  return "H4";
      case PERIOD_D1:  return "D1";
      case PERIOD_W1:  return "W1";
      case PERIOD_MN1: return "MN1";
      default:         return "UNKNOWN";
   }
}

//+------------------------------------------------------------------+
//| 錯誤描述                                                         |
//+------------------------------------------------------------------+
string ErrorDescription(int error)
{
   switch(error)
   {
      case 0:    return "No error";
      case 1:    return "No error, but result unknown";
      case 2:    return "Common error";
      case 3:    return "Invalid trade parameters";
      case 4:    return "Trade server is busy";
      case 5:    return "Old version of client terminal";
      case 6:    return "No connection with trade server";
      case 7:    return "Not enough rights";
      case 8:    return "Too frequent requests";
      case 9:    return "Malfunctional trade operation";
      case 64:   return "Account disabled";
      case 65:   return "Invalid account";
      case 128:  return "Trade timeout";
      case 129:  return "Invalid price";
      case 130:  return "Invalid stops";
      case 131:  return "Invalid trade volume";
      case 132:  return "Market is closed";
      case 133:  return "Trade is disabled";
      case 134:  return "Not enough money";
      case 135:  return "Price changed";
      case 136:  return "Off quotes";
      case 137:  return "Broker is busy";
      case 138:  return "Requote";
      case 139:  return "Order is locked";
      case 140:  return "Buy orders only allowed";
      case 141:  return "Too many requests";
      case 145:  return "Modification denied";
      case 146:  return "Trade context is busy";
      case 147:  return "Expiration denied";
      case 148:  return "Too many orders";
      default:   return "Unknown error: " + IntegerToString(error);
   }
}

//+------------------------------------------------------------------+
//| 清理資源                                                         |
//+------------------------------------------------------------------+
void Cleanup()
{
   isConnected = false;
   
   if(rep_socket != 0)
   {
      zmq_close(rep_socket);
      rep_socket = 0;
   }
   
   if(pub_socket != 0)
   {
      zmq_close(pub_socket);
      pub_socket = 0;
   }
   
   if(context != 0)
   {
      zmq_ctx_destroy(context);
      context = 0;
   }
}