import json
import requests

def translate_qa_report(screen, description, expected):
    url = "http://localhost:11434/api/generate"
    
    # Standardizing the QA input
    qa_input = {
        "screen_name": screen,
        "issue_description": description,
        "expected_outcome": expected
    }
    
    prompt = f"""
    Translate this QA bug report into developer Tanglish.
    
    Input JSON:
    {json.dumps(qa_input, indent=2)}
    
    Output JSON:
    """
    
    payload = {
        "model": "qwen-tanglish-translation:latest",
        "prompt": prompt,
        "format": "json", # Forces Ollama to output valid JSON
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        result_json = json.loads(response.json()['response'])
        return result_json
    except Exception as e:
        return f"Error processing translation: {str(e)}"

# --- TEST RUN ---
qa_screen = "Cart & Checkout Screen"
qa_desc = "When user clicks on 'Apply Coupon', the total amount changes to $0, but the loading spinner spins indefinitely and the checkout button becomes completely grayed out."
qa_expected = "The coupon should apply a 10% discount, update the final price correctly, stop the spinner, and allow the user to complete the checkout process."

translated_report = translate_qa_report(qa_screen, qa_desc, qa_expected)
# print(json.dumps(translated_report, indent=4))
print(translated_report)
