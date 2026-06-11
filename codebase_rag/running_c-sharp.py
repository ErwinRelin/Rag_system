import subprocess

# Run the C# project using the .NET CLI
result = subprocess.run(["dotnet", "run", "--project", r"C:\Users\Erwin\Desktop\rag_system\codebase_rag\DummyApplication\DummyApplication.csproj"], capture_output=True, text=True)

# Print the C# program output
print("Output from C#:")
print(result.stdout)
