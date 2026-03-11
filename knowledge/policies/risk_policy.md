# Risk Policy & Governance

## Risk Level Definitions

### LOW (0-30)

- **Description:** Normal market conditions
- **Action:** Monitor only
- **Escalation:** None required

### MEDIUM (31-60)

- **Description:** Elevated risk requiring attention
- **Action:** Review positions, consider hedging
- **Escalation:** Daily summary to risk team

### HIGH (61-80)

- **Description:** Significant risk event
- **Action:** Active risk management required
- **Escalation:** Real-time notification to portfolio managers
- **Compliance:** Requires dual confirmation (sentiment + volatility)

### CRITICAL (81-100)

- **Description:** Severe risk requiring immediate action
- **Action:** Emergency position review, potential de-risking
- **Escalation:** Immediate notification to CRO and senior leadership
- **Compliance:** Requires risk committee sign-off for any new positions

## Alert Frequency Rules

1. **Anti-Spam:** Maximum 1 alert per ticker per hour
2. **Escalation Path:** HIGH alerts → Slack; CRITICAL → Slack + PagerDuty
3. **Off-Hours:** CRITICAL alerts only outside market hours (9:30 AM - 4 PM ET)

## Data Quality Requirements

- **Minimum Data Points:** 10 news articles or 20 social posts for confident sentiment
- **Staleness Threshold:** Market data must be < 5 minutes old
- **Confidence Floor:** Sentiment confidence must be > 0.3 to trigger alerts
