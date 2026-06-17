using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;

class Program
{
    static void Main(string[] args)
    {
        // 1. Validate Input Directory
        if (args.Length == 0)
        {
            Console.Error.WriteLine("Error: Please provide the path to your C# codebase directory.");
            Console.Error.WriteLine("Usage: dotnet run -- <path_to_source_code>");
            return;
        }

        string targetDirectory = args[0];
        if (!Directory.Exists(targetDirectory))
        {
            Console.Error.WriteLine($"Error: Directory '{targetDirectory}' does not exist.");
            return;
        }

        // 2. Find all C# files
        var csFiles = Directory.GetFiles(targetDirectory, "*.cs", SearchOption.AllDirectories)
            .Where(file => IsValidDomainFile(file))
            .ToArray();

        if (csFiles.Length == 0)
        {
            Console.Error.WriteLine("No valid domain .cs files found.");
            return;
        }

        var callGraph = new HashSet<string>();

        var references = new List<MetadataReference>
        {
            MetadataReference.CreateFromFile(typeof(object).Assembly.Location),
            MetadataReference.CreateFromFile(Path.Combine(Path.GetDirectoryName(typeof(object).Assembly.Location)!, "System.Runtime.dll")),
            MetadataReference.CreateFromFile(Path.Combine(Path.GetDirectoryName(typeof(object).Assembly.Location)!, "System.Console.dll")),
            MetadataReference.CreateFromFile(Path.Combine(Path.GetDirectoryName(typeof(object).Assembly.Location)!, "System.Linq.dll")),
            MetadataReference.CreateFromFile(Path.Combine(Path.GetDirectoryName(typeof(object).Assembly.Location)!, "System.Linq.Expressions.dll"))
        };

        var syntaxTrees = new List<SyntaxTree>();
        foreach (var file in csFiles)
        {
            string code = File.ReadAllText(file);
            syntaxTrees.Add(CSharpSyntaxTree.ParseText(code, path: file));
        }

        string uniqueAssemblyName = $"GlobalWorkspace_{Guid.NewGuid():N}";
        var compilation = CSharpCompilation.Create(uniqueAssemblyName)
            .WithOptions(new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary))
            .AddReferences(references)
            .AddSyntaxTrees(syntaxTrees);

        var globalTypesCache = compilation.Assembly.GlobalNamespace.GetAllTypes().ToList();

        // 3. Process Syntax Trees
        foreach (var tree in syntaxTrees)
        {
            var semanticModel = compilation.GetSemanticModel(tree);
            var root = tree.GetRoot();

            ExtractRepositoryDeclarations(root, semanticModel, callGraph, tree.FilePath);

            var invocations = root.DescendantNodes().OfType<InvocationExpressionSyntax>();
            var objectCreations = root.DescendantNodes().OfType<ObjectCreationExpressionSyntax>();

            foreach (var invocation in invocations)
            {
                ProcessInvocationExpression(invocation, semanticModel, globalTypesCache, callGraph, tree.FilePath);
            }

        }

        // 4. Output deduplicated JSON
        Console.WriteLine("[");
        Console.WriteLine(string.Join(",\n", callGraph));
        Console.WriteLine("]");
    }

    private static bool IsValidDomainFile(string filePath)
    {
        string normalizedPath = filePath.Replace("\\", "/").ToLower();
        return !normalizedPath.Contains("/obj/") &&
               !normalizedPath.Contains("/bin/") &&
               !normalizedPath.Contains("/.vs/") &&
               !normalizedPath.Contains("/venv/") &&
               !normalizedPath.Contains("/qdrant_storage_db/") &&
               !normalizedPath.Contains("c-sharp_roslyn");
    }

    private static void ProcessInvocationExpression(InvocationExpressionSyntax invocation, SemanticModel semanticModel, List<INamedTypeSymbol> globalTypesCache, HashSet<string> callGraph, string filePath)
    {
        var symbolInfo = semanticModel.GetSymbolInfo(invocation);
        IMethodSymbol? targetMethod = symbolInfo.Symbol as IMethodSymbol;

        string structuralRelationOverride = "";
        string structuralTargetClass = "";
        string structuralTargetMethod = "";

        if (invocation.Expression is MemberAccessExpressionSyntax memberAccess)
        {
            string methodName = memberAccess.Name.Identifier.Text;
            string expressionsChain = memberAccess.ToString();

            if (expressionsChain.Contains("File.Write") || expressionsChain.Contains("writelines") || expressionsChain.Contains("Path.write_text"))
                return;

            bool isEFKeywords = methodName == "Include" || methodName == "ThenInclude" || 
                               methodName == "ToListAsync" || methodName == "FirstOrDefaultAsync" ||
                               methodName == "AsNoTracking" || methodName == "Where" || methodName == "Select";

            if (isEFKeywords || expressionsChain.Contains("Context.") || expressionsChain.Contains("DbSet"))
            {
                structuralRelationOverride = "DATABASE_QUERY";
                structuralTargetClass = "Microsoft.EntityFrameworkCore.DbContext";
                structuralTargetMethod = methodName;
            }
        }

        if (targetMethod != null)
        {
            ExtractRelationship(invocation, targetMethod, semanticModel, globalTypesCache, callGraph, filePath, structuralRelationOverride);
        }
        else if (!string.IsNullOrEmpty(structuralRelationOverride))
        {
            WriteStructuralFallbackEdge(invocation, structuralTargetClass, structuralTargetMethod, structuralRelationOverride, callGraph, filePath);
        }
    }

    private static bool IsFrameworkType(string typeName)
    {
        return
            typeName.StartsWith("System.") ||
            typeName.StartsWith("Microsoft.") ||

            typeName.Contains("System.Console") ||
            typeName.Contains("System.String") ||

            typeName.Contains("Enumerable") ||
            typeName.Contains("List<") ||
            typeName.Contains("Dictionary<") ||
            typeName.Contains("HashSet<");
    }

    private static void ExtractRelationship(SyntaxNode node, IMethodSymbol targetMethod, SemanticModel semanticModel, List<INamedTypeSymbol> globalTypesCache, HashSet<string> callGraph, string currentFilePath, string relationOverride)
    {
        INamedTypeSymbol? containingType = targetMethod.ContainingType;
        string targetClass = containingType?.ToDisplayString() ?? "UnknownClass";
        string targetName = targetMethod.MethodKind == MethodKind.Constructor ? ".ctor" : targetMethod.Name;

        if (IsFrameworkType(targetClass))
                return;

        if (targetMethod.MethodKind == MethodKind.Constructor)
                return;

        if (targetName == "WriteAllLines" || targetName == "WriteAllText" || targetClass.Contains("System.IO.File"))
            return;

        var sourceClassNode = node.Ancestors().OfType<ClassDeclarationSyntax>().FirstOrDefault();
        var sourceMethodNode = node.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();

        if (sourceClassNode != null && sourceMethodNode != null)
        {
            var sourceClassSymbol = semanticModel.GetDeclaredSymbol(sourceClassNode);
            string sourceClass = sourceClassSymbol?.ToDisplayString() ?? "UnknownClass";
            string sourceMethod = sourceMethodNode.Identifier.Text;

            string safePath = currentFilePath.Replace("\\", "/");
            var sourceLocation = sourceMethodNode.GetLocation().GetLineSpan();
            int startLine = sourceLocation.StartLinePosition.Line + 1;
            int endLine = sourceLocation.EndLinePosition.Line + 1;

            bool srcIsPartial = sourceClassNode.Modifiers.Any(m => m.IsKind(SyntaxKind.PartialKeyword));
            bool srcIsSealed = sourceClassNode.Modifiers.Any(m => m.IsKind(SyntaxKind.SealedKeyword));
            bool tgtIsPartial = containingType?.DeclaringSyntaxReferences.Length > 1;
            bool tgtIsSealed = containingType?.IsSealed ?? false;

            // 1. Resolve Interface Implementations
            if (containingType != null && (containingType.TypeKind == TypeKind.Interface || targetClass.Contains("Repository")))
            {
                var implementations = globalTypesCache
                    .Where(t => t.AllInterfaces.Contains(containingType, SymbolEqualityComparer.Default) || 
                                (t.BaseType != null && SymbolEqualityComparer.Default.Equals(t.BaseType, containingType)));

                bool resolved = false;
                foreach (var implType in implementations)
                {
                    var implMethod = implType.FindImplementationForInterfaceMember(targetMethod) as IMethodSymbol;
                    string actualMethodName = implMethod != null ? implMethod.Name : targetName;
                    string implClassName = implType.ToDisplayString();

                    string stepRelation = (implClassName.Contains("Repository") || implClassName.Contains(".Repositories.")) 
                                          ? "REPOSITORY_ACCESS" : "IMPLEMENTATION_CALL";

                    resolved = true;
                    AddJsonEdge(sourceClass, sourceMethod, srcIsPartial, srcIsSealed, 
                                implClassName, actualMethodName, implType.DeclaringSyntaxReferences.Length > 1, implType.IsSealed,
                                stepRelation, safePath, startLine, endLine, callGraph);
                }

                if (resolved) return;
            }

            // 2. Identify Relation Types
            string finalRelation = relationOverride ?? "";

                if (string.IsNullOrEmpty(finalRelation))
                {
                    string targetTypeName = containingType?.Name ?? "";

                    if (targetClass.Contains("Microsoft.EntityFrameworkCore") ||
                        targetClass.Contains("DbContext"))
                    {
                        finalRelation = "DATABASE_QUERY";
                    }
                    else if (targetTypeName.EndsWith("Repository"))
                    {
                        finalRelation = "REPOSITORY_ACCESS";
                    }
                    else if (targetTypeName.EndsWith("Service"))
                    {
                        finalRelation = "SERVICE_ACCESS";
                    }
                    else
                    {
                        finalRelation = "DIRECT_CALL";
                    }
                }

            // 3. DI Bindings
            if (targetName.StartsWith("AddScoped") || targetName.StartsWith("AddTransient") || targetName.StartsWith("AddSingleton"))
            {
                if (targetMethod.IsGenericMethod && targetMethod.TypeArguments.Length == 2)
                {
                    string interfaceType = targetMethod.TypeArguments[0].ToDisplayString();
                    string concreteType = targetMethod.TypeArguments[1].ToDisplayString();
                    
                    AddJsonEdge(interfaceType, "DI_REGISTERED_AS", false, false, 
                                concreteType, "DependencyInjection", false, false, 
                                "DI_BINDING", safePath, startLine, endLine, callGraph);
                }
            }

            AddJsonEdge(sourceClass, sourceMethod, srcIsPartial, srcIsSealed, 
                        targetClass, targetName, tgtIsPartial, tgtIsSealed, 
                        finalRelation, safePath, startLine, endLine, callGraph);

            
        }
    }

    private static void WriteStructuralFallbackEdge(InvocationExpressionSyntax invocation, string targetClass, string targetMethod, string relation, HashSet<string> callGraph, string currentFilePath)
    {
        var sourceClassNode = invocation.Ancestors().OfType<ClassDeclarationSyntax>().FirstOrDefault();
        var sourceMethodNode = invocation.Ancestors().OfType<MethodDeclarationSyntax>().FirstOrDefault();

        if (sourceClassNode != null && sourceMethodNode != null)
        {
            string sourceClass = sourceClassNode.Identifier.Text;
            string sourceMethod = sourceMethodNode.Identifier.Text;
            var loc = sourceMethodNode.GetLocation().GetLineSpan();
            
            AddJsonEdge(sourceClass, sourceMethod, false, false,
                        targetClass, targetMethod, false, false,
                        relation, currentFilePath.Replace("\\", "/"), loc.StartLinePosition.Line + 1, loc.EndLinePosition.Line + 1, callGraph);
        }
    }

    private static void ExtractRepositoryDeclarations(SyntaxNode root, SemanticModel semanticModel, HashSet<string> callGraph, string currentFilePath)
    {
        var classes = root.DescendantNodes().OfType<ClassDeclarationSyntax>();
        foreach (var classNode in classes)
        {
            var symbol = semanticModel.GetDeclaredSymbol(classNode);
            if (symbol == null) continue;

            string className = symbol.ToDisplayString();
            if (className.Contains("Repository") || (symbol.BaseType?.Name.Contains("DbContext") ?? false))
            {
                var loc = classNode.GetLocation().GetLineSpan();
                AddJsonEdge(className, "ClassDeclaration", false, false,
                            "DatabaseCluster.Infrastructure", "StorageEngine", false, true,
                            "DB_REPOSITORY_SCHEMA", currentFilePath.Replace("\\", "/"), loc.StartLinePosition.Line + 1, loc.EndLinePosition.Line + 1, callGraph);
            }
        }
    }

    private static void AddJsonEdge(string srcClass, string srcMethod, bool srcPartial, bool srcSealed,
                                    string tgtClass, string tgtMethod, bool tgtPartial, bool tgtSealed,
                                    string relType, string filePath, int startLine, int endLine, HashSet<string> callGraph)
    {
        // Using C# 11 Raw String Literal for clean JSON formatting
        string jsonBlock = $$"""
          {
            "source_class": "{{srcClass}}",
            "source_method": "{{srcMethod}}",
            "source_is_partial": {{srcPartial.ToString().ToLower()}},
            "source_is_sealed": {{srcSealed.ToString().ToLower()}},
            "target_class": "{{tgtClass}}",
            "target_method": "{{tgtMethod}}",
            "target_is_partial": {{tgtPartial.ToString().ToLower()}},
            "target_is_sealed": {{tgtSealed.ToString().ToLower()}},
            "relation_type": "{{relType}}",
            "file_path": "{{filePath}}",
            "chunk_start_line": {{startLine}},
            "chunk_end_line": {{endLine}}
          }
        """;
        callGraph.Add(jsonBlock);
    }
}

public static class SymbolExtensions
{
    public static IEnumerable<INamedTypeSymbol> GetAllTypes(this INamespaceSymbol namespaceSymbol)
    {
        foreach (var type in namespaceSymbol.GetTypeMembers())
        {
            yield return type;
            foreach (var nestedType in type.GetNestedTypes())
                yield return nestedType;
        }

        foreach (var nestedNamespace in namespaceSymbol.GetNamespaceMembers())
            foreach (var type in nestedNamespace.GetAllTypes())
                yield return type;
    }

    public static IEnumerable<INamedTypeSymbol> GetNestedTypes(this INamedTypeSymbol typeSymbol)
    {
        foreach (var nested in typeSymbol.GetTypeMembers())
        {
            yield return nested;
            foreach (var deeperNested in nested.GetNestedTypes())
                yield return deeperNested;
        }
    }
}