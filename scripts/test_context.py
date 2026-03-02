import requests
import time

def test_context():
    base_url = "http://localhost:8000"
    session_id = f"test_session_{int(time.time())}"
    
    print(f"Starting test with session_id: {session_id}")
    
    # Step 1: Establish context
    payload1 = {
        "situation": "My son Alex is staying up until 3 AM every night playing games.",
        "session_id": session_id
    }
    print("\nSending message 1...")
    resp1 = requests.post(f"{base_url}/chat", json=payload1)
    resp1.raise_for_status()
    print("Response 1:", resp1.json()["response"])
    
    # Wait for context derivation (it's synchronous in this implementation but good practice)
    time.sleep(1)
    
    # Step 2: Follow up with shorthand
    payload2 = {
        "situation": "He also does this on school nights, and it makes me very frustrated.",
        "session_id": session_id
    }
    print("\nSending message 2 (follow-up)...")
    resp2 = requests.post(f"{base_url}/chat", json=payload2)
    resp2.raise_for_status()
    response2_text = resp2.json()["response"]
    print("Response 2:", response2_text)
    
    # Check if PACE remembers 'Alex' or the 'late night' context
    mentions_alex = "alex" in response2_text.lower()
    mentions_gaming = "gam" in response2_text.lower() or "night" in response2_text.lower()
    
    if mentions_alex or mentions_gaming:
        print("\nSUCCESS: PACE seems to remember the context!")
    else:
        print("\nWARNING: PACE might not have linked the context clearly.")

if __name__ == "__main__":
    test_context()
