import os
import requests

API_URL = "http://127.0.0.1:5000/predict"

# Define local relative paths to 3 of your sample TIFF inputs
SAMPLES = [
    "sample_input_0.tif",
    "sample_input_1.tif",
    "sample_input_2.tif"
]

def test_api_with_file(file_path):
    if not os.path.exists(file_path):
        print(f"[-] Skip: File '{file_path}' does not exist locally.")
        return False
        
    print(f"[+] Uploading '{file_path}' to /predict...")
    with open(file_path, 'rb') as f:
        response = requests.post(API_URL, files={'file': f})
        
    if response.status_code == 200:
        data = response.json()
        print(f"[✓] Status: {response.status_code} | OK")
        print(f"    - Returned Mask Base64 Length: {len(data['mask_image'])} chars")
        print(f"    - Returned Preview Base64 Length: {len(data['rgb_preview'])} chars\n")
        return True
    else:
        print(f"[x] Status: {response.status_code} | Error")
        print(f"    - Detail: {response.text}\n")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Programmatic API Validation (3 Sample Checks)")
    print("="*60)
    
    successful_tests = 0
    for sample in SAMPLES:
        if test_api_with_file(sample):
            successful_tests += 1
            
    print("="*60)
    print(f"Execution complete. Passed {successful_tests}/{len(SAMPLES)} tests.")
    print("="*60)