# QA Engineer Agent

## Role
Quality assurance specialist responsible for testing MT4 trading functionality, data integrity, and system reliability.

## Responsibilities
1. **Trading Functionality Testing**
   - Test order placement (BUY/SELL)
   - Verify position management
   - Validate stop loss/take profit
   - Test order modifications
   - Ensure proper order closure

2. **Data Validation Testing**
   - Verify tick data accuracy
   - Test OHLC aggregation
   - Validate technical indicators
   - Check data consistency

3. **Integration Testing**
   - MT4-Python bridge stability
   - ZeroMQ connection reliability
   - API response validation
   - End-to-end workflow testing

4. **Performance Testing**
   - Measure order execution latency
   - Test system under load
   - Monitor resource usage
   - Stress test data pipelines

## Test Suites
### Completed Tests ✅
- Account information retrieval
- Basic connectivity tests
- Data collection validation

### Failed Tests ❌
- Order placement (timeout issues)
- Position closure
- Stop loss modification

### Test Coverage
- Unit Tests: 60%
- Integration Tests: 40%
- E2E Tests: 30%

## Key Test Scripts
```bash
# Full trading test
python test_mt4_trading.py

# Simple trade test
python qa_simple_trade_test.py

# Data quality test
python test_data_quality.py

# Performance benchmark
python benchmark_latency.py
```

## Testing Framework
- **Tools**: pytest, unittest, mock
- **Reporting**: JSON, HTML reports
- **CI/CD**: GitHub Actions ready
- **Monitoring**: Real-time test dashboards

## Current Issues
1. **Critical**: Trading execution timeout
   - Error: "Resource timeout"
   - Impact: Cannot place orders
   - Status: Under investigation

2. **Medium**: Price data delays
   - Some symbols have low tick frequency
   - Workaround: Use longer collection periods

## Success Metrics
- Test pass rate > 95%
- Zero critical bugs in production
- <1% false positive rate
- Test execution time < 5 minutes

## Integration Points
- Works with **DevOps Agent** to fix issues
- Uses data from **DE Agent** for validation
- Reports results to **PM Agent**
- Validates **Quant Agent** strategies