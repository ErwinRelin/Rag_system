import json
import os
import tempfile
import sys
import subprocess

def apply_codebase_changes_with_logging(json_data):
    data = json.loads(json_data)
    
    if data.get("status") != "SUCCESS":
        print(f"❌ Aborting: JSON status is '{data.get('status')}', not 'SUCCESS'.")
        return

    print(f"🔍 Found {len(data['changes'])} total task(s) to process.")

    for idx, change in enumerate(data["changes"], 1):
        file_path = change["file_path"]
        operation = change["operation"]
        target_scope = change["target_scope"]
        anchor = change["anchor"]
        new_code = change["new_code"]
        
        print(f"\n--- Task {idx}: {operation} on {target_scope} ---")
        print(f"📂 Checking path: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ Error: Python cannot locate this file on disk! Check folder path permissions.")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        modified_lines = []
        insert_index = -1
        
        # --- PROCESSING APPEND_TO_CLASS ---
        if operation == "APPEND_TO_CLASS":
            found_class = False
            brace_count = 0
            
            for line_idx, line in enumerate(lines):
                if f"class {target_scope}" in line:
                    found_class = True
                if found_class:
                    brace_count += line.count("{")
                    brace_count -= line.count("}")
                    if brace_count == 0 and "}" in line:
                        insert_index = line_idx
                        break
            
            if insert_index != -1:
                clean_code = new_code.strip()
                padded_code = f"        {clean_code}\n"
                lines.insert(insert_index, padded_code)
                modified_lines = lines
                print(f"✅ Matched class boundary! Ready to insert at line {insert_index + 1}.")
            else:
                print(f"❌ Error: Could not parse brace structure for class '{target_scope}'.")
                modified_lines = lines

        # --- PROCESSING INSERT_AFTER METHOD BOUNDARY ---
        elif operation == "INSERT_AFTER":
            found_anchor = False
            brace_count = 0
            method_started = False

            for line_idx, line in enumerate(lines):
                if anchor in line:
                    found_anchor = True
                
                if found_anchor:
                    if "{" in line:
                        brace_count += line.count("{")
                        method_started = True
                    if "}" in line:
                        brace_count -= line.count("}")
                    
                    if method_started and brace_count == 0:
                        insert_index = line_idx + 1
                        break
            
            if insert_index != -1:
                clean_method = new_code.strip('\n')
                lines.insert(insert_index, f"\n{clean_method}\n")
                modified_lines = lines
                print(f"✅ Found closing brace of method! Ready to insert at line {insert_index + 1}.")
            else:
                print(f"❌ Error: Could not locate method signature string: '{anchor}'")
                modified_lines = lines

        # Write to temp file and overwrite safely
        if insert_index != -1:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.writelines(modified_lines)

            try:
                os.replace(temp_path, file_path)
                print(f"🎉 Successfully modified and updated file: {os.path.basename(file_path)}")
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                print(f"❌ File Swap Failure: {e}")

def verify_dotnet_syntax():
    print("\n Starting .NET multi-project syntax verification...")
    
    # This is the exact PowerShell script compressed into a single-line execution string
    ps_script = (
        "Get-ChildItem -Recurse -Filter *.csproj | ForEach-Object { "
        "Write-Host 'Checking Syntax:' $_.Name; "
        "dotnet build $_.FullName --no-incremental /p:BuildProjectReferences=false /v:quiet "
        "}"
    )
    
    # We call powershell.exe and pass the script securely via the -Command argument
    result = subprocess.run(
        ["powershell.exe", "-Command", ps_script],
        capture_output=True,
        text=True
    )
    
    # Print the terminal outputs so you can see any compilation syntax errors
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(f"System/Shell Errors:\n{result.stderr}")
        
    # Check if the execution context encountered errors
    if "error CS" in result.stdout or result.returncode != 0:
        print("❌ Syntax verification failed! Errors found in the codebase.")
        return False
    else:
        print("✅ Syntax verification passed! All projects are clean.")
        return True

# ==============================================================================
# FIXED: All Windows backslashes are completely escaped below (\\\\)
# ==============================================================================

if __name__ == "__main__":


    change_request = r"""
    {
    "status": "SUCCESS",
    "changes": [
        {
        "file_path": "C:/Users/Erwin/Desktop/rag_system/codebase_rag/DummyApplication/EmployeeService.cs",
        "target_scope": "EmployeeService",
        "operation": "APPEND_TO_CLASS",
        "anchor": "",
        "new_code": "public List<Employee> GetEmployeesWithConfiguredEmail()\n        {\n            return _employees.Where(e => !string.IsNullOrEmpty(e.Email)).ToList();\n        }"
        }
    ]
    }
    """

    apply_codebase_changes_with_logging(change_request)

    success = verify_dotnet_syntax()

    if not success:
        sys.exit(1)
