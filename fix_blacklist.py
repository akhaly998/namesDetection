#!/usr/bin/env python3
"""
Quick fix script to remove old BLACKLIST references
"""

import re

def fix_blacklist_references():
    """Fix remaining BLACKLIST references in main.py"""
    
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace old BLACKLIST references with blacklist_manager calls
    replacements = [
        (r'len\(BLACKLIST\)', 'len(blacklist_manager.get_blacklist())'),
        (r'BLACKLIST', 'blacklist_manager.get_blacklist()'),
        (r'for blacklisted_name in blacklist_manager\.get_blacklist\(\)\.get_blacklist\(\):', 
         'for blacklisted_name in blacklist_manager.get_blacklist():'),
    ]
    
    for old, new in replacements:
        content = re.sub(old, new, content)
    
    # Remove any duplicate endpoint definitions
    # Find and remove old duplicate get_blacklist endpoint
    pattern = r'@app\.get\("/blacklist"\)\nasync def get_blacklist\(\):[^@]+'
    content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
    
    with open('main.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Fixed BLACKLIST references in main.py")

if __name__ == "__main__":
    fix_blacklist_references()