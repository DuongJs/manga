# API Key Setup Guide

This guide explains how to configure Gemini API keys for the Manga Translator application.

## Problem

When running the application without proper API key configuration, you may see errors like:
```
Invalid Gemini API key. Please provide a valid API key from https://makersuite.google.com/app/apikey
```

This happens because the application cannot find a valid Gemini API key to use.

## Solutions

There are three ways to provide your Gemini API key:

### Option 1: Using the Web Interface (Recommended)

1. Run the application: `python app.py`
2. Open your browser and navigate to the local URL (typically `http://127.0.0.1:7860`)
3. Go to the "API Key Management" tab
4. Enter your Gemini API key in the "API Key" field
5. Optionally, give it a name (e.g., "Personal Key")
6. Click "Add API Key"

The key will be automatically saved to `api_keys.json` and will be available for future sessions.

### Option 2: Manually Create api_keys.json

1. Copy the example file:
   ```bash
   cp api_keys.json.example api_keys.json
   ```

2. Edit `api_keys.json` and replace `YOUR_GEMINI_API_KEY_HERE` with your actual API key:
   ```json
   {
     "api_keys": [
       {
         "key": "AIzaSy...",  // Your actual API key here
         "name": "Primary Key",
         "added_at": "2024-10-18T00:00:00.000000",
         "usage_count": 0
       }
     ],
     "current_index": 0,
     "last_updated": "2024-10-18T00:00:00.000000"
   }
   ```

3. Save the file and run the application

### Option 3: Environment Variable

Set the `GEMINI_API_KEY` environment variable:

**Linux/Mac:**
```bash
export GEMINI_API_KEY='your-api-key-here'
python app.py
```

**Windows (Command Prompt):**
```cmd
set GEMINI_API_KEY=your-api-key-here
python app.py
```

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="your-api-key-here"
python app.py
```

### Option 4: Enter API Key in UI

When using Gemini translation, you can also enter the API key directly in the "Gemini API Key" field in the translation interface.

## Getting Your API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key" or "Get API Key"
4. Copy the generated API key

## Multiple API Keys

The application supports multiple API keys with automatic round-robin rotation:

1. Add multiple keys through the web interface or manually in `api_keys.json`
2. The application will automatically rotate between keys
3. This helps with rate limiting and high-volume usage

Example with multiple keys:
```json
{
  "api_keys": [
    {
      "key": "AIzaSy...",
      "name": "Personal Key",
      "added_at": "2024-10-18T00:00:00.000000",
      "usage_count": 0
    },
    {
      "key": "AIzaSy...",
      "name": "Work Key",
      "added_at": "2024-10-18T01:00:00.000000",
      "usage_count": 0
    }
  ],
  "current_index": 0,
  "last_updated": "2024-10-18T00:00:00.000000"
}
```

## Troubleshooting

### "No API key file found"
- The `api_keys.json` file doesn't exist
- Solution: Create it using Option 2 above, or add keys via the web interface

### "Invalid JSON" error
- The `api_keys.json` file is malformed
- Solution: Check the file format against `api_keys.json.example`
- Common issues: Missing commas, quotes, or brackets

### "No API keys available in the manager"
- The `api_keys.json` file exists but contains no keys, or the keys array is empty
- Solution: Add keys using the web interface or manually edit the file

### API key works but translation fails
- Your API key may not have access to the Gemini API
- Your API quota may be exhausted
- Solution: Check your API key status at [Google AI Studio](https://makersuite.google.com/app/apikey)

## Security Note

⚠️ **Important**: Never commit `api_keys.json` to version control. This file is already in `.gitignore` to prevent accidental commits.
