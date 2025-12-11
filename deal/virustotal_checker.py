#!/usr/bin/env python3
"""
VirusTotal Benign Sample Checker
Automatically checks samples from dataset/benign against VirusTotal API
Respects daily quota of 500 lookups and saves results to dataset/virustotal
"""

import os
import json
import hashlib
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
API_KEY = "b279dc92c33aabd03b1dab4a3cf64398382f3fef8f5cfff8cb1c21229c0bd0d1"
DAILY_QUOTA = 500
BENIGN_DIR = Path(__file__).parent.parent.parent / "dataset" / "benign"
RESULTS_DIR = Path(__file__).parent.parent.parent / "dataset" / "virustotal"
PROGRESS_FILE = RESULTS_DIR / "progress.json"
RATE_LIMIT_DELAY = 15  # seconds between requests (4 requests per minute for free tier)

# VirusTotal API endpoints
VT_API_BASE = "https://www.virustotal.com/api/v3"
VT_FILE_ENDPOINT = f"{VT_API_BASE}/files"


class VirusTotalChecker:
    """Handles VirusTotal API interactions and result management"""
    
    def __init__(self):
        self.api_key = API_KEY
        self.headers = {
            "x-apikey": self.api_key,
            "Accept": "application/json"
        }
        self.results_dir = RESULTS_DIR
        self.progress_file = PROGRESS_FILE
        self.progress = self._load_progress()
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_progress(self) -> Dict:
        """Load progress from previous runs"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "checked_files": {},
            "last_run_date": None,
            "total_checked": 0
        }
    
    def _save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
    
    def calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def check_file_hash(self, file_hash: str) -> Optional[Dict]:
        """Check a file hash against VirusTotal"""
        url = f"{VT_FILE_ENDPOINT}/{file_hash}"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {"error": "File not found in VirusTotal database"}
            elif response.status_code == 429:
                print("⚠️  Rate limit exceeded. Waiting...")
                time.sleep(60)
                return self.check_file_hash(file_hash)
            else:
                return {"error": f"API error: {response.status_code}"}
        
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def save_result(self, filename: str, file_hash: str, result: Dict):
        """Save individual file result"""
        result_file = self.results_dir / f"{filename}.json"
        
        output = {
            "filename": filename,
            "sha256": file_hash,
            "check_date": datetime.now().isoformat(),
            "virustotal_result": result
        }
        
        with open(result_file, 'w') as f:
            json.dump(output, f, indent=2)
    
    def generate_summary(self):
        """Generate a summary report of all checked files"""
        summary = {
            "total_checked": self.progress["total_checked"],
            "last_run": datetime.now().isoformat(),
            "results": {
                "benign": 0,
                "malicious": 0,
                "suspicious": 0,
                "not_found": 0,
                "errors": 0
            },
            "details": []
        }
        
        for filename, data in self.progress["checked_files"].items():
            result_file = self.results_dir / f"{filename}.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                vt_result = result.get("virustotal_result", {})
                
                if "error" in vt_result:
                    if "not found" in vt_result["error"].lower():
                        summary["results"]["not_found"] += 1
                        status = "not_found"
                    else:
                        summary["results"]["errors"] += 1
                        status = "error"
                else:
                    stats = vt_result.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                    malicious = stats.get("malicious", 0)
                    suspicious = stats.get("suspicious", 0)
                    
                    if malicious > 0:
                        summary["results"]["malicious"] += 1
                        status = "malicious"
                    elif suspicious > 0:
                        summary["results"]["suspicious"] += 1
                        status = "suspicious"
                    else:
                        summary["results"]["benign"] += 1
                        status = "benign"
                
                summary["details"].append({
                    "filename": filename,
                    "sha256": data["sha256"],
                    "status": status,
                    "check_date": data["check_date"]
                })
        
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    def run_daily_check(self):
        """Run daily check with quota limit"""
        today = datetime.now().date().isoformat()
        
        # Check if we already ran today
        if self.progress["last_run_date"] == today:
            print(f"✓ Already ran today ({today}). Skipping to avoid quota waste.")
            print(f"Total files checked so far: {self.progress['total_checked']}")
            self.generate_summary()
            return
        
        # Get all files in benign directory
        all_files = sorted([f for f in BENIGN_DIR.glob("*.exe")])
        total_files = len(all_files)
        
        print(f"📁 Found {total_files} files in benign directory")
        print(f"✓ Already checked: {self.progress['total_checked']} files")
        print(f"📊 Daily quota: {DAILY_QUOTA} lookups")
        print(f"⏱️  Rate limit: {RATE_LIMIT_DELAY}s between requests")
        print("-" * 60)
        
        # Filter out already checked files
        files_to_check = [
            f for f in all_files 
            if f.name not in self.progress["checked_files"]
        ]
        
        if not files_to_check:
            print("✓ All files have been checked!")
            self.generate_summary()
            return
        
        # Limit to daily quota
        files_to_check = files_to_check[:DAILY_QUOTA]
        
        print(f"🔍 Checking {len(files_to_check)} files today...")
        print()
        
        checked_today = 0
        
        for idx, file_path in enumerate(files_to_check, 1):
            filename = file_path.name
            
            print(f"[{idx}/{len(files_to_check)}] {filename}")
            
            # Calculate hash
            print("  ⏳ Calculating SHA256...")
            file_hash = self.calculate_sha256(file_path)
            print(f"  🔑 Hash: {file_hash}")
            
            # Check with VirusTotal
            print("  🌐 Querying VirusTotal...")
            result = self.check_file_hash(file_hash)
            
            # Parse result
            if "error" in result:
                print(f"  ⚠️  {result['error']}")
            else:
                stats = result.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                malicious = stats.get("malicious", 0)
                suspicious = stats.get("suspicious", 0)
                harmless = stats.get("harmless", 0)
                
                if malicious > 0:
                    print(f"  ⚠️  MALICIOUS: {malicious} engines detected as malicious")
                elif suspicious > 0:
                    print(f"  ⚠️  SUSPICIOUS: {suspicious} engines flagged as suspicious")
                else:
                    print(f"  ✓ BENIGN: {harmless} engines marked as harmless")
            
            # Save result
            self.save_result(filename, file_hash, result)
            
            # Update progress
            self.progress["checked_files"][filename] = {
                "sha256": file_hash,
                "check_date": datetime.now().isoformat()
            }
            self.progress["total_checked"] += 1
            checked_today += 1
            
            # Save progress after each file
            self._save_progress()
            
            print(f"  💾 Saved to {filename}.json")
            print()
            
            # Rate limiting (except for last file)
            if idx < len(files_to_check):
                print(f"  ⏸️  Waiting {RATE_LIMIT_DELAY}s (rate limit)...")
                time.sleep(RATE_LIMIT_DELAY)
                print()
        
        # Update last run date
        self.progress["last_run_date"] = today
        self._save_progress()
        
        # Generate summary
        print("-" * 60)
        print("📊 Generating summary report...")
        summary = self.generate_summary()
        
        print()
        print("=" * 60)
        print("✓ DAILY CHECK COMPLETE")
        print("=" * 60)
        print(f"Checked today: {checked_today} files")
        print(f"Total checked: {self.progress['total_checked']} / {total_files} files")
        print(f"Remaining: {total_files - self.progress['total_checked']} files")
        print()
        print("Results Summary:")
        print(f"  ✓ Benign: {summary['results']['benign']}")
        print(f"  ⚠️  Malicious: {summary['results']['malicious']}")
        print(f"  ⚠️  Suspicious: {summary['results']['suspicious']}")
        print(f"  ❓ Not Found: {summary['results']['not_found']}")
        print(f"  ❌ Errors: {summary['results']['errors']}")
        print()
        print(f"Results saved to: {self.results_dir}")
        print(f"Summary report: {self.results_dir / 'summary.json'}")
        print("=" * 60)


def main():
    """Main entry point"""
    print("=" * 60)
    print("VirusTotal Benign Sample Checker")
    print("=" * 60)
    print()
    
    checker = VirusTotalChecker()
    checker.run_daily_check()


if __name__ == "__main__":
    main()
