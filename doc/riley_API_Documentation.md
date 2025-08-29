# API Documentation - Riley AI Intelligence & Memory Module

This document provides the API design for the Riley AI Intelligence & Memory Module, designed to integrate with a primary voice assistant (e.g., Lisa).

## Base URL

`https://api.riley.ai/v1` (or `http://localhost:8001/v1` for local development)

## Authentication

All API requests must be authenticated using an API key. The API key should be included in the `Authorization` header of the request.

`Authorization: Bearer <YOUR_API_KEY>`

## Endpoints

### 1. Process Intelligence Request

*   **Endpoint:** `POST /intelligence/process`
*   **Description:** Receives transcribed text and conversational context from the primary voice assistant, processes it using Riley's intelligence and memory, and returns an intelligent response.
*   **Request Body:**

```json
{
  "user_id": "<unique-user-identifier>",
  "transcribed_text": "<text-from-STT>",
  "conversation_history": [
    {"role": "user", "content": "<previous-user-utterance>"},
    {"role": "assistant", "content": "<previous-assistant-response>"}
  ],
  "current_emotion": "<optional-detected-emotion>"
}
```

*   **Response Body:**

```json
{
  "response_text": "<riley-generated-intelligent-response>",
  "memory_updated": true,
  "new_emotion_state": "<optional-new-emotion-state>"
}
```

### 2. Get User Memory History

*   **Endpoint:** `GET /memory/history`
*   **Description:** Retrieves a summary of the user's long-term memory or interaction history managed by Riley.
*   **Query Parameters:**
    *   `user_id` (required): The unique identifier for the user.
    *   `limit` (optional): The maximum number of records to return.
    *   `offset` (optional): The number of records to skip.
*   **Response Body:**

```json
{
  "user_id": "<unique-user-identifier>",
  "history_summary": [
    {
      "timestamp": "<timestamp>",
      "event_type": "<memory-event-type>",
      "description": "<summary-of-memory-event>"
    }
  ]
}
```

### 3. Get System Status

*   **Endpoint:** `GET /system/status`
*   **Description:** Checks the operational status of Riley's intelligence and memory components.
*   **Response Body:**

```json
{
  "status": "ok",
  "components": {
    "natural-language-understanding": "ok",
    "memory-database": "ok",
    "emotion-analysis": "ok",
    "response-generation": "ok"
  }
}
```