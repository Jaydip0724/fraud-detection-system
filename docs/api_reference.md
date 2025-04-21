# API Reference

## Overview

The Fraud Detection System provides a RESTful API for monitoring model performance, retrieving transaction data, and accessing system statistics. All API endpoints return data in JSON format.

## Base URL

```
http://[hostname]:8000
```

## Authentication

Currently, the API does not require authentication. This is suitable for internal use only. For production deployment, appropriate authentication mechanisms should be implemented.

## Endpoints

### System Status

```
GET /
```

Returns the current system status and server time.

#### Response

```json
{
  "status": "running",
  "time": "2025-04-21T15:30:45.123456"
}
```

#### Status Codes

- `200 OK`: System is running properly

---

### Model Metrics

```
GET /metrics
```

Returns the latest performance metrics of the fraud detection model.

#### Response

```json
{
  "precision": 0.9245,
  "recall": 0.8732,
  "f1": 0.8982,
  "accuracy": 0.9967
}
```

If no metrics are available yet:

```json
{
  "precision": 0.0,
  "recall": 0.0,
  "f1": 0.0,
  "accuracy": 0.0,
  "message": "No metrics available yet"
}
```

#### Field Descriptions

- `precision`: Percentage of transactions flagged as fraud that are actually fraudulent
- `recall`: Percentage of actual fraudulent transactions that are correctly identified
- `f1`: Harmonic mean of precision and recall
- `accuracy`: Overall accuracy of the model

#### Status Codes

- `200 OK`: Metrics retrieved successfully
- `500 Internal Server Error`: Error retrieving metrics

---

### Feature Statistics

```
GET /statistics
```

Returns statistical information about transaction features, including means and standard deviations.

#### Response

```json
{
  "statistics": [
    {
      "feature": "v1",
      "mean": 0.0012,
      "std": 1.9867
    },
    {
      "feature": "v2",
      "mean": -0.0034,
      "std": 1.6514
    },
    ...
    {
      "feature": "amount",
      "mean": 88.3496,
      "std": 250.1201
    }
  ]
}
```

#### Field Descriptions

- `feature`: Feature name
- `mean`: Average value of the feature
- `std`: Standard deviation of the feature

#### Status Codes

- `200 OK`: Statistics retrieved successfully
- `500 Internal Server Error`: Error retrieving statistics

---

### Recent Transactions

```
GET /transactions
```

Returns the 100 most recent transactions processed by the system, including their classification results.

#### Response

```json
{
  "transactions": [
    {
      "time": "2025-04-21T15:30:45.123456",
      "amount": 123.45,
      "actual_class": 0,
      "predicted_class": 0
    },
    {
      "time": "2025-04-21T15:25:12.654321",
      "amount": 5000.00,
      "actual_class": 1,
      "predicted_class": 1
    },
    ...
  ]
}
```

#### Field Descriptions

- `time`: Timestamp of the transaction
- `amount`: Transaction amount
- `actual_class`: Actual classification (0 = normal, 1 = fraud)
- `predicted_class`: Model prediction (0 = normal, 1 = fraud)

#### Status Codes

- `200 OK`: Transactions retrieved successfully
- `500 Internal Server Error`: Error retrieving transactions

---

## Error Responses

All endpoints may return the following error response format:

```json
{
  "error": "Error message description"
}
```

### Common Error Codes

- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server-side error

## Rate Limiting

Currently, there are no rate limits imposed on the API. However, excessive requests may impact system performance.

## Data Freshness

- `/metrics` and `/statistics` endpoints return the most recently calculated values, which are updated with each processed transaction.
- The `/transactions` endpoint returns the 100 most recent transactions in reverse chronological order.

## Notes on Deployment

When deploying in a production environment, consider the following:

1. Implement proper authentication
2. Set up HTTPS
3. Configure appropriate CORS policies
4. Implement rate limiting
5. Add more granular error handling

## API Versioning

The current API version is v1. All endpoints are currently unversioned, but future API versions will be prefixed with `/v2/`, `/v3/`, etc.
