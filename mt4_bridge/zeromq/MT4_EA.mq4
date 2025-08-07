//+------------------------------------------------------------------+
//|                                              ZeroMQ_Bridge_EA.mq4 |
//|                                       Python-MT4 Bridge EA        |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "QuantProject"
#property version   "1.00"
#property strict

// 包含 ZeroMQ 庫
#include <Zmq/Zmq.mqh>

// ZeroMQ 設置
input string InpZmqPullAddress = "tcp://*:5555";  // 接收 Python 命令
input string InpZmqPushAddress = "tcp://*:5556";  // 發送數據到 Python

// 全局變量
Context context;
Socket pullSocket(context, ZMQ_PULL);
Socket pushSocket(context, ZMQ_PUSH);

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
    // 綁定套接字
    if (!pullSocket.bind(InpZmqPullAddress)) {
        Print("Failed to bind pull socket");
        return INIT_FAILED;
    }
    
    if (!pushSocket.bind(InpZmqPushAddress)) {
        Print("Failed to bind push socket");
        return INIT_FAILED;
    }
    
    Print("ZeroMQ Bridge EA initialized successfully");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    pullSocket.unbind(InpZmqPullAddress);
    pushSocket.unbind(InpZmqPushAddress);
    Print("ZeroMQ Bridge EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    // 檢查是否有來自 Python 的命令
    string message = "";
    
    if (pullSocket.recv(message, ZMQ_DONTWAIT)) {
        ProcessCommand(message);
    }
}

//+------------------------------------------------------------------+
//| Process command from Python                                       |
//+------------------------------------------------------------------+
void ProcessCommand(string jsonMessage)
{
    // 解析 JSON 命令
    // 注意：MT4 原生不支持 JSON，這裡簡化處理
    // 實際使用需要 JSON 解析庫
    
    if (StringFind(jsonMessage, "PLACE_ORDER") >= 0) {
        ProcessPlaceOrder(jsonMessage);
    }
    else if (StringFind(jsonMessage, "CLOSE_ORDER") >= 0) {
        ProcessCloseOrder(jsonMessage);
    }
    else if (StringFind(jsonMessage, "GET_QUOTE") >= 0) {
        ProcessGetQuote(jsonMessage);
    }
    else if (StringFind(jsonMessage, "GET_ACCOUNT_INFO") >= 0) {
        ProcessGetAccountInfo();
    }
    else if (StringFind(jsonMessage, "GET_POSITIONS") >= 0) {
        ProcessGetPositions();
    }
    else if (StringFind(jsonMessage, "GET_HISTORY") >= 0) {
        ProcessGetHistory(jsonMessage);
    }
}

//+------------------------------------------------------------------+
//| Place order                                                       |
//+------------------------------------------------------------------+
void ProcessPlaceOrder(string message)
{
    // 簡化的參數解析（實際需要 JSON 解析）
    string symbol = "EURUSD";  // 從 message 解析
    int orderType = OP_BUY;    // 從 message 解析
    double volume = 0.01;      // 從 message 解析
    double price = 0;          // 市價
    double sl = 0;
    double tp = 0;
    
    // 執行下單
    int ticket = OrderSend(symbol, orderType, volume, 
                          orderType == OP_BUY ? Ask : Bid, 
                          3, sl, tp, "Python Bridge", 0, 0, clrGreen);
    
    // 發送結果到 Python
    string response;
    if (ticket > 0) {
        response = StringFormat("{\"success\": true, \"ticket\": %d}", ticket);
    } else {
        response = StringFormat("{\"success\": false, \"error\": %d}", GetLastError());
    }
    
    pushSocket.send(response);
}

//+------------------------------------------------------------------+
//| Get quote                                                         |
//+------------------------------------------------------------------+
void ProcessGetQuote(string message)
{
    // 從 message 解析 symbol
    string symbol = "EURUSD";
    
    double bid = MarketInfo(symbol, MODE_BID);
    double ask = MarketInfo(symbol, MODE_ASK);
    double spread = MarketInfo(symbol, MODE_SPREAD);
    
    string response = StringFormat(
        "{\"symbol\": \"%s\", \"bid\": %.5f, \"ask\": %.5f, \"spread\": %.0f}",
        symbol, bid, ask, spread
    );
    
    pushSocket.send(response);
}

//+------------------------------------------------------------------+
//| Get account info                                                  |
//+------------------------------------------------------------------+
void ProcessGetAccountInfo()
{
    string response = StringFormat(
        "{\"balance\": %.2f, \"equity\": %.2f, \"margin\": %.2f, \"free_margin\": %.2f, \"leverage\": %d}",
        AccountBalance(),
        AccountEquity(),
        AccountMargin(),
        AccountFreeMargin(),
        AccountLeverage()
    );
    
    pushSocket.send(response);
}

//+------------------------------------------------------------------+
//| Get open positions                                                |
//+------------------------------------------------------------------+
void ProcessGetPositions()
{
    string positions = "[";
    bool first = true;
    
    for (int i = 0; i < OrdersTotal(); i++) {
        if (OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) {
            if (!first) positions += ",";
            first = false;
            
            positions += StringFormat(
                "{\"ticket\": %d, \"symbol\": \"%s\", \"type\": %d, \"volume\": %.2f, \"price\": %.5f, \"profit\": %.2f}",
                OrderTicket(),
                OrderSymbol(),
                OrderType(),
                OrderLots(),
                OrderOpenPrice(),
                OrderProfit()
            );
        }
    }
    
    positions += "]";
    pushSocket.send(positions);
}

//+------------------------------------------------------------------+
//| Get history data                                                  |
//+------------------------------------------------------------------+
void ProcessGetHistory(string message)
{
    // 從 message 解析參數
    string symbol = "EURUSD";
    int timeframe = PERIOD_H1;
    int bars = 100;
    
    string history = "[";
    
    for (int i = 0; i < bars && i < Bars; i++) {
        if (i > 0) history += ",";
        
        history += StringFormat(
            "{\"time\": %d, \"open\": %.5f, \"high\": %.5f, \"low\": %.5f, \"close\": %.5f, \"volume\": %d}",
            iTime(symbol, timeframe, i),
            iOpen(symbol, timeframe, i),
            iHigh(symbol, timeframe, i),
            iLow(symbol, timeframe, i),
            iClose(symbol, timeframe, i),
            iVolume(symbol, timeframe, i)
        );
    }
    
    history += "]";
    pushSocket.send(history);
}

//+------------------------------------------------------------------+
//| Close order                                                       |
//+------------------------------------------------------------------+
void ProcessCloseOrder(string message)
{
    // 從 message 解析 ticket
    int ticket = 12345;  // 實際需要從 message 解析
    
    bool success = false;
    if (OrderSelect(ticket, SELECT_BY_TICKET)) {
        double price = OrderType() == OP_BUY ? Bid : Ask;
        success = OrderClose(ticket, OrderLots(), price, 3, clrRed);
    }
    
    string response = StringFormat("{\"success\": %s}", success ? "true" : "false");
    pushSocket.send(response);
}