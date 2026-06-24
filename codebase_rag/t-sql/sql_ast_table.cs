using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using Microsoft.SqlServer.TransactSql.ScriptDom;

namespace SqlChunkerApp
{
    // ─────────────────────────────────────────────────────────────────────────
    // Represents one FK relationship with full column-level detail
    // ─────────────────────────────────────────────────────────────────────────
    class FkRelationship
    {
        public string LocalColumn   { get; set; }  // e.g. "EmployeeId"
        public string LocalType     { get; set; }  // e.g. "INT NOT NULL"
        public string TargetTable   { get; set; }  // e.g. "Employees"
        public string TargetColumn  { get; set; }  // e.g. "EmployeeId"
        public string ConstraintName { get; set; } // e.g. "FK_Employee_Department" (if named)

        // Human-readable join hint the LLM can use directly
        public override string ToString() =>
            $"{LocalColumn} → {TargetTable}.{TargetColumn}" +
            (string.IsNullOrEmpty(ConstraintName) ? "" : $"  [constraint: {ConstraintName}]");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Represents one enriched chunk ready to be passed to an LLM or vector DB
    // ─────────────────────────────────────────────────────────────────────────
    class SqlChunk
{
    public int                  ChunkId          { get; set; }
    public string               FileName         { get; set; }
    public string               ObjectType       { get; set; }
    public string               ChunkCategory    { get; set; }
    public string               ObjectName       { get; set; }
    public string               NlDescription    { get; set; }
    public List<FkRelationship> FkRelationships  { get; set; } = new();
    public string               SqlText          { get; set; }
    public string               FullContextBlock { get; set; }
    public int                  ReferencingDegree { get; set; }  // NEW: FK referencing degree
    public string               EntityType       { get; set; }  // NEW: "Core", "Supporting", or "Junction"

    public string ReferencedTables =>
        string.Join(", ", FkRelationships.Select(r => r.TargetTable).Distinct());
}

    class Program
    {

        static void ClassifyEntityTypes(List<SqlChunk> ddlChunks, 
                         Dictionary<string, int> referencingDegrees,
                         double coreThresholdPercent = 20.0,
                         int junctionMinIncoming = 1,
                         int junctionMaxOutgoing = 2)
        {
            var tables = ddlChunks.Where(c => c.ObjectType == "CreateTableStatement").ToList();
            
            if (!tables.Any()) return;
            
            var referencingValues = tables.Select(t => 
                referencingDegrees.ContainsKey(t.ObjectName) ? referencingDegrees[t.ObjectName] : 0
            ).ToList();
            
            double avgReferencingDegree = referencingValues.Average();
            double maxReferencingDegree = referencingValues.Max();
            
            double masterThreshold = Math.Max(avgReferencingDegree * 2, maxReferencingDegree * 0.7);
            double supportingThreshold = Math.Max(avgReferencingDegree * 0.5, 1);
            
            foreach (var table in tables)
            {
                int referencingDegree = referencingDegrees.ContainsKey(table.ObjectName) 
                    ? referencingDegrees[table.ObjectName] 
                    : 0;
                int outgoingFks = table.FkRelationships.Count;
                
                table.ReferencingDegree = referencingDegree;
                
                // Priority 1: Check if it's a junction table
                if (IsJunctionTable(table, referencingDegree, outgoingFks, junctionMinIncoming, junctionMaxOutgoing) 
                    && !HasBusinessData(table))  // Don't classify business tables as junctions
                {
                    table.EntityType = "Junction";
                }
                // Priority 2: Check if it's a master entity
                else if (referencingDegree >= masterThreshold || (outgoingFks == 0 && referencingDegree >= 3))
                {
                    table.EntityType = "Master";
                }
                // Priority 3: Check if it's a transaction entity
                else if (outgoingFks >= 1 && HasBusinessData(table))
                {
                    table.EntityType = "Transaction";
                }
                // Priority 4: Check if it's a supporting entity (lookup/reference table)
                else if ((outgoingFks == 0 && referencingDegree >= 1) || IsSupportingEntity(table, tables))
                {
                    table.EntityType = "Supporting";  // Fixed: was "Reference"
                }
                // Priority 5: Everything else is a leaf
                else
                {
                    table.EntityType = "Leaf";
                }
            }
        }

        

        static bool HasBusinessData(SqlChunk table)
        {
            var sqlUpper = table.SqlText.ToUpper();
            
            // Check for business columns
            bool hasDates = sqlUpper.Contains("DATE") || sqlUpper.Contains("TIME");
            bool hasStatus = sqlUpper.Contains("STATUS") || sqlUpper.Contains("STATE");
            bool hasAmounts = sqlUpper.Contains("DECIMAL") || sqlUpper.Contains("AMOUNT") || 
                            sqlUpper.Contains("SALARY") || sqlUpper.Contains("PRICE") || 
                            sqlUpper.Contains("TOTAL") || sqlUpper.Contains("QUANTITY") ||
                            sqlUpper.Contains("ALLOWANCE") || sqlUpper.Contains("DEDUCTION");
            bool hasDescriptions = sqlUpper.Contains("VARCHAR") && 
                                (sqlUpper.Contains("NAME") || sqlUpper.Contains("DESCRIPTION") || 
                                sqlUpper.Contains("NOTE") || sqlUpper.Contains("COMMENT"));
            
            // Count non-FK columns
            int nonFkCols = CountNonFkColumns(table);
            
            return nonFkCols >= 3 || hasDates || hasStatus || hasAmounts || hasDescriptions;
        }

        static int CountNonFkColumns(SqlChunk table)
        {
            int count = 0;
            var sqlLines = table.SqlText.Split('\n');
            
            foreach (var line in sqlLines)
            {
                var trimmed = line.Trim().ToUpper();
                
                if (trimmed.Length == 0 || 
                    trimmed.StartsWith("CREATE") || 
                    trimmed.StartsWith("GO") ||
                    trimmed.StartsWith("CONSTRAINT") ||
                    trimmed.StartsWith("FOREIGN") ||
                    trimmed.StartsWith("PRIMARY") ||
                    trimmed.StartsWith(")") ||
                    trimmed.StartsWith("("))
                    continue;
                
                bool isColumn = trimmed.Contains("INT") || trimmed.Contains("VARCHAR") || 
                            trimmed.Contains("DECIMAL") || trimmed.Contains("DATE") || 
                            trimmed.Contains("BIT") || trimmed.Contains("CHAR") ||
                            trimmed.Contains("TIME");
                
                if (isColumn)
                {
                    bool isFkColumn = table.FkRelationships.Any(fk => 
                        trimmed.TrimStart().StartsWith(fk.LocalColumn.ToUpper()));
                    
                    if (!isFkColumn)
                    {
                        count++;
                    }
                }
            }
            
            return count;
        }

        static bool IsSupportingEntity(SqlChunk table, List<SqlChunk> allTables)
        {
            int outgoingFks = table.FkRelationships.Count;
            int referencingDegree = table.ReferencingDegree;
            
            // Supporting entities are typically lookup/reference tables
            if (outgoingFks == 0 && referencingDegree >= 1)
            {
                string[] supportingPatterns = { "Type", "Status", "Category", "Period", "Title", "Level", "Grade", "Code", "Role", "Group" };
                bool nameSuggestsSupporting = supportingPatterns.Any(p => 
                    table.ObjectName.IndexOf(p, StringComparison.OrdinalIgnoreCase) >= 0);
                
                // If it has 0 FKs and is referenced, it's at least Supporting
                return true;
            }
            
            return false;
        }

        static bool IsJunctionTable(SqlChunk table, int referencingDegree, int outgoingFks,
                            int minIncoming, int maxOutgoing)
        {
            // Junction tables typically:
            // 1. Have exactly 2 foreign keys pointing to other tables
            // 2. Are referenced by few or no other tables
            // 3. Don't have much business data beyond FKs
            
            bool hasFewOutgoingFks = outgoingFks >= 2 && outgoingFks <= maxOutgoing;
            bool hasFewIncomingRefs = referencingDegree <= minIncoming;
            
            // Check if table name suggests junction (common naming patterns)
            string[] junctionNamePatterns = { "Map", "Mapping", "Link", "Junction", "Assoc", "Bridge", "Xref", "Cross" };
            bool nameSuggestsJunction = junctionNamePatterns.Any(p => 
                table.ObjectName.IndexOf(p, StringComparison.OrdinalIgnoreCase) >= 0);
            
            // Check if the table is primarily FKs (few business columns)
            int nonFkColumns = CountNonFkColumns(table);
            bool isPrimarilyFKs = nonFkColumns <= 2;  // Junction tables have very few non-FK columns
            
            // Must NOT have business data
            bool hasNoBusinessData = !HasBusinessData(table);
            
            return hasFewOutgoingFks && hasFewIncomingRefs && hasNoBusinessData && (nameSuggestsJunction || isPrimarilyFKs);
        }

        static Dictionary<string, int> CalculateReferencingDegree(List<SqlChunk> ddlChunks)
        {
            var referencingCount = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
            
            foreach (var chunk in ddlChunks.Where(c => c.ObjectType == "CreateTableStatement"))
            {
                foreach (var fk in chunk.FkRelationships)
                {
                    if (!referencingCount.ContainsKey(fk.TargetTable))
                        referencingCount[fk.TargetTable] = 0;
                    referencingCount[fk.TargetTable]++;
                }
            }
            
            return referencingCount;
        }

        static void Main(string[] args)
        {
            string targetFolder = @"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql";

            if (!Directory.Exists(targetFolder))
            {
                Console.WriteLine($"Error: Folder does not exist at '{targetFolder}'");
                return;
            }

            string[] sqlFiles = Directory.GetFiles(targetFolder, "*.sql");
            Console.WriteLine($"Found {sqlFiles.Length} SQL file(s) to process.\n");

            var parser = new TSql160Parser(initialQuotedIdentifiers: true);
            var options = new SqlScriptGeneratorOptions
            {
                SqlVersion            = SqlVersion.Sql160,
                KeywordCasing         = KeywordCasing.Uppercase,
                NewLineBeforeFromClause = true,
                NewLineBeforeJoinClause = true
            };
            var generator = new Sql160ScriptGenerator(options);

            foreach (string filePath in sqlFiles)
            {
                string fileName = Path.GetFileName(filePath);
                Console.WriteLine($"**************************************************");
                Console.WriteLine($"STARTING FILE: {fileName}");
                Console.WriteLine($"**************************************************\n");

                using var reader = new StreamReader(filePath);
                var fragment = parser.Parse(reader, out IList<ParseError> errors);

                if (errors.Count > 0)
                {
                    Console.WriteLine($"⚠️  Skipped '{fileName}'. Found {errors.Count} syntax error(s):");
                    foreach (var e in errors)
                        Console.WriteLine($"   Line {e.Line}: {e.Message}");
                    Console.WriteLine();
                    continue;
                }

                if (fragment is not TSqlScript script) continue;

                // ── Pass 1: collect all chunks ────────────────────────────────
                var allChunks  = new List<SqlChunk>();
                var ddlChunks  = new List<SqlChunk>(); // DDL only, for schema summary
                int chunkId    = 1;

                foreach (TSqlBatch batch in script.Batches)
                {
                    foreach (TSqlStatement statement in batch.Statements)
                    {
                        generator.GenerateScript(statement, out string rawSql);
                        rawSql = rawSql.Trim();

                        string objectType = statement.GetType().Name;
                        string category   = ClassifyStatement(objectType);
                        string objectName = ExtractObjectName(statement);
                        var    fkRels     = ExtractForeignKeyRelationships(statement);
                        string nlDesc     = BuildNlDescription(statement, objectType, objectName,
                                                               fkRels, category);

                        var chunk = new SqlChunk
                        {
                            ChunkId         = chunkId++,
                            FileName        = fileName,
                            ObjectType      = objectType,
                            ChunkCategory   = category,
                            ObjectName      = objectName,
                            NlDescription   = nlDesc,
                            FkRelationships = fkRels,
                            SqlText         = rawSql
                        };

                        allChunks.Add(chunk);
                        if (category is "DDL" or "VIEW" or "PROCEDURE")
                            ddlChunks.Add(chunk);
                    }
                }

                // In Main method, after collecting all chunks:

                var referencingDegrees = CalculateReferencingDegree(ddlChunks);

                // After assigning referencing degrees, add this to update NL descriptions:
                foreach (var chunk in ddlChunks.Where(c => c.ObjectType == "CreateTableStatement"))
                {
                    if (!string.IsNullOrEmpty(chunk.EntityType))
                    {
                        string oldClassification = DetermineEntityTypeDescription(chunk.FkRelationships, chunk.ObjectName);
                        string newClassification = GetEntityTypeDescription(chunk.EntityType);
                        
                        chunk.NlDescription = chunk.NlDescription.Replace(
                            $"Entity Classification: {oldClassification}",
                            $"Entity Classification: {newClassification}");
                    }
                }

                static string GetEntityTypeDescription(string entityType)
                {
                    return entityType switch
                    {
                        "Master" => "Master entity — central to the domain and heavily referenced",
                        "Transaction" => "Transaction entity — represents business events/processes",
                        "Supporting" => "Supporting entity — provides lookup/reference data",
                        "Junction" => "Junction entity — manages many-to-many relationships",
                        "Leaf" => "Leaf entity — standalone with minimal relationships",
                        _ => entityType
                    };
                }

                ClassifyEntityTypes(ddlChunks, referencingDegrees, 
                    coreThresholdPercent: 20.0,
                    junctionMinIncoming: 1, 
                    junctionMaxOutgoing: 2);

                foreach (var chunk in ddlChunks)
                {
                    if (chunk.ObjectType == "CreateTableStatement")
                    {
                        chunk.ReferencingDegree = referencingDegrees.ContainsKey(chunk.ObjectName) 
                            ? referencingDegrees[chunk.ObjectName] 
                            : 0;
                    }
                }

                // ── Pass 2: build the schema summary chunk ────────────────────
                SqlChunk summaryChunk = BuildSchemaSummaryChunk(fileName, ddlChunks);

                // ── Pass 3: build full context blocks and emit ────────────────
                foreach (SqlChunk chunk in allChunks)
                {
                    chunk.FullContextBlock = BuildFullContextBlock(chunk);
                    EmitChunk(chunk);
                }

                // Emit the summary last (or pass it as a system-level chunk)
                EmitChunk(summaryChunk);

                Console.WriteLine($"Finished processing file: {fileName}\n\n");
            }
        }

        // ─────────────────────────────────────────────────────────────────────
        // Categorise statement type into a coarse bucket
        // ─────────────────────────────────────────────────────────────────────
        static string ClassifyStatement(string objectType) => objectType switch
        {
            "CreateTableStatement"     => "DDL",
            "CreateDatabaseStatement"  => "DDL",
            "UseStatement"             => "DDL",
            "AlterTableStatement"      => "DDL",
            "CreateIndexStatement"     => "DDL",
            "CreateViewStatement"      => "VIEW",
            "CreateProcedureStatement" => "PROCEDURE",
            "CreateFunctionStatement"  => "PROCEDURE",
            "InsertStatement"          => "SEED_DATA",
            "UpdateStatement"          => "DML",
            "DeleteStatement"          => "DML",
            "SelectStatement"          => "DML",
            _                          => "OTHER"
        };

        // ─────────────────────────────────────────────────────────────────────
        // Best-effort extraction of the primary object name from a statement
        // ─────────────────────────────────────────────────────────────────────
        static string ExtractObjectName(TSqlStatement stmt) => stmt switch
        {
            CreateTableStatement     s => s.SchemaObjectName?.BaseIdentifier?.Value ?? "",
            CreateViewStatement      s => s.SchemaObjectName?.BaseIdentifier?.Value ?? "",
            CreateProcedureStatement s => s.ProcedureReference?.Name?.BaseIdentifier?.Value ?? "",
            CreateFunctionStatement  s => s.Name?.BaseIdentifier?.Value ?? "",
            CreateDatabaseStatement  s => s.DatabaseName?.Value ?? "",
            _                          => ""
        };

        // ─────────────────────────────────────────────────────────────────────
        // Walk FK constraints and return a structured list with full column detail
        // ─────────────────────────────────────────────────────────────────────
        static List<FkRelationship> ExtractForeignKeyRelationships(TSqlStatement stmt)
        {
            var result = new List<FkRelationship>();
            if (stmt is not CreateTableStatement createTable) return result;

            // Build a lookup: column name → data type string, for annotation
            var colTypeMap = createTable.Definition.ColumnDefinitions.ToDictionary(
                c => c.ColumnIdentifier.Value,
                c => DescribeColumnType(c)
            );

            // ── Table-level FK constraints (CONSTRAINT FK_x FOREIGN KEY ...) ──
            foreach (var constraint in createTable.Definition.TableConstraints)
            {
                if (constraint is not ForeignKeyConstraintDefinition fk) continue;

                string targetTable  = fk.ReferenceTableName?.BaseIdentifier?.Value ?? "";
                string constraintName = (constraint as ConstraintDefinition)?.ConstraintIdentifier?.Value ?? "";

                // Pair each local column with its corresponding target column by position
                for (int i = 0; i < fk.Columns.Count; i++)
                {
                    string localCol  = fk.Columns[i]?.Value ?? "";
                    string targetCol = i < fk.ReferencedTableColumns.Count
                        ? fk.ReferencedTableColumns[i]?.Value ?? ""
                        : "";
                    colTypeMap.TryGetValue(localCol, out string localType);

                    result.Add(new FkRelationship
                    {
                        LocalColumn    = localCol,
                        LocalType      = localType ?? "",
                        TargetTable    = targetTable,
                        TargetColumn   = targetCol,
                        ConstraintName = constraintName
                    });
                }
            }

            // ── Column-level FK constraints (inline REFERENCES ...) ──
            foreach (var col in createTable.Definition.ColumnDefinitions)
            {
                foreach (var constraint in col.Constraints)
                {
                    if (constraint is not ForeignKeyConstraintDefinition fk) continue;

                    string localCol   = col.ColumnIdentifier.Value;
                    string targetTable = fk.ReferenceTableName?.BaseIdentifier?.Value ?? "";
                    string targetCol  = fk.ReferencedTableColumns.FirstOrDefault()?.Value ?? "";
                    colTypeMap.TryGetValue(localCol, out string localType);

                    result.Add(new FkRelationship
                    {
                        LocalColumn    = localCol,
                        LocalType      = localType ?? "",
                        TargetTable    = targetTable,
                        TargetColumn   = targetCol,
                        ConstraintName = ""
                    });
                }
            }

            return result;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Produce a concise type string for a column, e.g. "INT NOT NULL"
        // ─────────────────────────────────────────────────────────────────────
        static string DescribeColumnType(ColumnDefinition col)
        {
            string typeName = col.DataType switch
            {
                SqlDataTypeReference dt => dt.SqlDataTypeOption.ToString().ToUpper(),
                _ => col.DataType?.GetType().Name ?? "UNKNOWN"
            };

            if (col.DataType is SqlDataTypeReference sqlDt && sqlDt.Parameters.Count > 0)
            {
                string param = string.Join(", ", sqlDt.Parameters.Select(p =>
                {
                    if (p is IntegerLiteral il) return il.Value;
                    if (p is MaxLiteral) return "MAX";
                    return p.ToString();
                }));
                typeName += $"({param})";
            }

            // Check for NOT NULL constraints
            bool notNull = false;
            
            // Check column-level constraints for NullableConstraintDefinition
            foreach (var constraint in col.Constraints)
            {
                if (constraint is NullableConstraintDefinition nullableConstraint)
                {
                    notNull = !nullableConstraint.Nullable;
                    break;
                }
            }
            
            // Check for unique constraints that might be primary keys
            if (!notNull)
            {
                notNull = col.Constraints.Any(c => 
                    c is UniqueConstraintDefinition unique && unique.IsPrimaryKey);
            }
            
            // If no explicit NULL/NOT NULL constraint and no PRIMARY KEY, 
            // check if the column has IDENTITY (IDENTITY columns are typically NOT NULL)
            if (!notNull)
            {
                notNull = col.IdentityOptions != null;
            }

            typeName += notNull ? " NOT NULL" : " NULL";

            return typeName;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Produce a plain-English description with inline per-column FK detail
        // ─────────────────────────────────────────────────────────────────────
        static string BuildNlDescription(TSqlStatement stmt, string objectType,
                                 string objectName, List<FkRelationship> fkRels,
                                 string category)
        {
            if (category == "SEED_DATA")
                return "Sample / seed data INSERT — not schema definition. " +
                    "Exclude this chunk when only schema context is needed.";

            if (category == "DDL" && objectType == "CreateDatabaseStatement")
                return $"Creates the top-level database named '{objectName}'.";

            if (category == "DDL" && objectType == "UseStatement")
                return "Switches the active database context.";

            if (stmt is CreateTableStatement createTable)
            {
                // Build a lookup so each column can be annotated with its FK target inline
                var fkByLocalCol = fkRels.ToDictionary(r => r.LocalColumn, r => r);

                var sb = new StringBuilder();
                sb.AppendLine($"Defines the '{objectName}' table. Columns:");

                foreach (var col in createTable.Definition.ColumnDefinitions)
                {
                    string colName = col.ColumnIdentifier.Value;
                    string colType = DescribeColumnType(col);

                    if (fkByLocalCol.TryGetValue(colName, out FkRelationship fk))
                        // Inline the FK target right next to the column it belongs to
                        sb.AppendLine($"  • {colName} ({colType})  →  joins {fk.TargetTable}.{fk.TargetColumn}" +
                                    (string.IsNullOrEmpty(fk.ConstraintName) ? "" : $" [{fk.ConstraintName}]"));
                    else
                        sb.AppendLine($"  • {colName} ({colType})");
                }

                if (!fkRels.Any())
                    sb.AppendLine("  No foreign-key dependencies.");

                // Add entity type classification
                string entityType = DetermineEntityTypeDescription(fkRels, objectName);
                sb.AppendLine();
                sb.AppendLine($"  Entity Classification: {entityType}");

                return sb.ToString().TrimEnd();
            }

            if (stmt is CreateViewStatement)
                return $"View '{objectName}' — a pre-built SELECT that joins multiple tables " +
                    "and can be queried directly without rewriting the join logic.";

            if (stmt is CreateProcedureStatement)
                return $"Stored procedure '{objectName}' — encapsulates business logic that " +
                    "can be called by name; inspect the body for parameters and DML operations.";

            return $"{objectType} statement on object '{objectName}'.";
        }

        // Helper method - moved outside BuildNlDescription
        static string DetermineEntityTypeDescription(List<FkRelationship> fkRels, string tableName)
        {
            if (!fkRels.Any())
                return "Standalone table with no foreign key dependencies";
            
            if (fkRels.Count == 2)
                return "Potential junction/bridge table connecting two entities";
            
            if (fkRels.Count == 1)
                return "References a single parent table";
            
            return $"References {fkRels.Count} parent tables";
        }

        // ─────────────────────────────────────────────────────────────────────
        // Assemble a single schema-summary chunk for the whole file
        // ─────────────────────────────────────────────────────────────────────
        static SqlChunk BuildSchemaSummaryChunk(string fileName, List<SqlChunk> ddlChunks)
        {
            var sb = new StringBuilder();
            sb.AppendLine("=== SCHEMA SUMMARY ===");
            sb.AppendLine($"File: {fileName}");
            sb.AppendLine();

            var tables = ddlChunks.Where(c => c.ObjectType == "CreateTableStatement").ToList();
            var views  = ddlChunks.Where(c => c.ObjectType == "CreateViewStatement").ToList();
            var procs  = ddlChunks.Where(c => c.ObjectType == "CreateProcedureStatement").ToList();

            if (tables.Any())
            {
                sb.AppendLine("TABLES:");
                foreach (var t in tables)
                {
                    sb.AppendLine($"  • {t.ObjectName}");
                    sb.AppendLine($"    {t.NlDescription}");
                    if (t.FkRelationships.Any())
                    {
                        sb.AppendLine($"    Join paths:");
                        foreach (var fk in t.FkRelationships)
                            sb.AppendLine($"      {fk}");
                    }
                }
                sb.AppendLine();
            }

            if (views.Any())
            {
                sb.AppendLine("VIEWS:");
                foreach (var v in views)
                    sb.AppendLine($"  • {v.ObjectName}: {v.NlDescription}");
                sb.AppendLine();
            }

            if (procs.Any())
            {
                sb.AppendLine("STORED PROCEDURES:");
                foreach (var p in procs)
                    sb.AppendLine($"  • {p.ObjectName}: {p.NlDescription}");
                sb.AppendLine();
            }

            // Relationship map — one line per FK, with full column detail
            sb.AppendLine("RELATIONSHIP MAP:");
            foreach (var t in tables.Where(t => t.FkRelationships.Any()))
            {
                sb.AppendLine($"  {t.ObjectName}:");
                foreach (var fk in t.FkRelationships)
                    sb.AppendLine($"    {fk}");
            }

            var referencingDegrees = CalculateReferencingDegree(ddlChunks);

            sb.AppendLine("REFERENCING DEGREES:");
            foreach (var t in tables)
            {
                int degree = referencingDegrees.ContainsKey(t.ObjectName) 
                    ? referencingDegrees[t.ObjectName] 
                    : 0;
                sb.AppendLine($"  • {t.ObjectName}: referenced by {degree} table(s)");
            }

            // In BuildSchemaSummaryChunk method, add after FK DEGREES section:

            // In BuildSchemaSummaryChunk, replace the ENTITY CLASSIFICATION section:

            sb.AppendLine("ENTITY CLASSIFICATION:");
            sb.AppendLine();

            var masterEntities = tables.Where(t => t.EntityType == "Master").ToList();
            var transactionEntities = tables.Where(t => t.EntityType == "Transaction").ToList();
            var supportingEntities = tables.Where(t => t.EntityType == "Supporting").ToList();
            var junctionEntities = tables.Where(t => t.EntityType == "Junction").ToList();
            var leafEntities = tables.Where(t => t.EntityType == "Leaf").ToList();

            if (masterEntities.Any())
            {
                sb.AppendLine("  MASTER ENTITIES (central to domain, heavily referenced):");
                foreach (var entity in masterEntities)
                {
                    sb.AppendLine($"    • {entity.ObjectName} (referenced by {entity.ReferencingDegree} tables)");
                    sb.AppendLine($"      Outgoing FKs: {entity.FkRelationships.Count}");
                }
                sb.AppendLine();
            }

            if (transactionEntities.Any())
            {
                sb.AppendLine("  TRANSACTION ENTITIES (business events/processes):");
                foreach (var entity in transactionEntities)
                {
                    sb.AppendLine($"    • {entity.ObjectName}");
                    sb.AppendLine($"      References: {string.Join(", ", entity.FkRelationships.Select(fk => fk.TargetTable))}");
                    sb.AppendLine($"      Referenced by: {entity.ReferencingDegree} tables");
                }
                sb.AppendLine();
            }

            if (supportingEntities.Any())
            {
                sb.AppendLine("  SUPPORTING ENTITIES (lookup/reference data):");
                foreach (var entity in supportingEntities)
                {
                    sb.AppendLine($"    • {entity.ObjectName} (referenced by {entity.ReferencingDegree} tables)");
                    sb.AppendLine($"      Outgoing FKs: {entity.FkRelationships.Count}");
                }
                sb.AppendLine();
            }

            if (junctionEntities.Any())
            {
                sb.AppendLine("  JUNCTION ENTITIES (many-to-many relationships):");
                foreach (var entity in junctionEntities)
                {
                    sb.AppendLine($"    • {entity.ObjectName}");
                    sb.AppendLine($"      Connects: {string.Join(", ", entity.FkRelationships.Select(fk => fk.TargetTable))}");
                }
                sb.AppendLine();
            }

            if (leafEntities.Any())
            {
                sb.AppendLine("  LEAF ENTITIES (standalone with minimal relationships):");
                foreach (var entity in leafEntities)
                {
                    sb.AppendLine($"    • {entity.ObjectName} (referenced by {entity.ReferencingDegree} tables)");
                }
                sb.AppendLine();
            }

            // In BuildSchemaSummaryChunk, replace the ENTITY RELATIONSHIPS section with:

            // In BuildSchemaSummaryChunk, update the ENTITY RELATIONSHIPS section:

            sb.AppendLine("ENTITY RELATIONSHIPS:");
            foreach (var entity in tables)
            {
                sb.Append($"  [{entity.EntityType}] {entity.ObjectName}");
                if (entity.ReferencingDegree > 0 && entity.FkRelationships.Any())
                    sb.Append($" ← referenced by {entity.ReferencingDegree} tables");
                else if (entity.ReferencingDegree > 0)
                    sb.Append($" ← referenced by {entity.ReferencingDegree} tables");
                else if (entity.FkRelationships.Any())
                    sb.Append(" → references other tables");
                else
                    sb.Append(" (standalone)");
                sb.AppendLine();
                
                if (entity.FkRelationships.Any())
                {
                    foreach (var fk in entity.FkRelationships)
                    {
                        var targetEntity = tables.FirstOrDefault(t => 
                            t.ObjectName.Equals(fk.TargetTable, StringComparison.OrdinalIgnoreCase));
                        string targetType = targetEntity?.EntityType ?? "Unknown";
                        sb.AppendLine($"    └─→ [{targetType}] {fk.TargetTable}.{fk.TargetColumn}");
                    }
                }
                else if (entity.ReferencingDegree == 0 && !entity.FkRelationships.Any())
                {
                    // Find tables that reference this one
                    var referencingTables = tables.Where(t => 
                        t.FkRelationships.Any(fk => 
                            fk.TargetTable.Equals(entity.ObjectName, StringComparison.OrdinalIgnoreCase)));
                    if (referencingTables.Any())
                    {
                        sb.AppendLine($"    (referenced by: {string.Join(", ", referencingTables.Select(t => t.ObjectName))})");
                    }
                }
            }

            string summaryText = sb.ToString().Trim();

            return new SqlChunk
            {
                ChunkId          = 0,   // 0 = schema summary, always prepend in retrieval
                FileName         = fileName,
                ObjectType       = "SchemaSummary",
                ChunkCategory    = "SCHEMA_SUMMARY",
                ObjectName       = fileName,
                NlDescription    = "High-level overview of every table, view, and procedure in this file, including FK relationships. Always include this chunk in LLM context.",
                FkRelationships  = new(),
                SqlText          = summaryText,
                FullContextBlock = summaryText
            };
        }

        // ─────────────────────────────────────────────────────────────────────
        // Build the full string that gets passed to the LLM API
        // ─────────────────────────────────────────────────────────────────────
        static string BuildFullContextBlock(SqlChunk chunk)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"[CHUNK]");
            sb.AppendLine($"  File     : {chunk.FileName}");
            sb.AppendLine($"  Id       : {chunk.ChunkId}");
            sb.AppendLine($"  Category : {chunk.ChunkCategory}");
            sb.AppendLine($"  Object   : {chunk.ObjectName}");
            
            // Add entity classification for tables
            if (chunk.ObjectType == "CreateTableStatement" && !string.IsNullOrEmpty(chunk.EntityType))
            {
                sb.AppendLine($"  Entity Type : {chunk.EntityType}");
                sb.AppendLine($"  Referenced By : {chunk.ReferencingDegree} table(s)");
            }

            if (chunk.FkRelationships.Any())
            {
                sb.AppendLine($"  FK Joins :");
                foreach (var fk in chunk.FkRelationships)
                    sb.AppendLine($"    {fk}");
            }

            sb.AppendLine($"  Summary  : {chunk.NlDescription}");
            sb.AppendLine();
            sb.AppendLine("[SQL]");
            sb.AppendLine(chunk.SqlText);
            sb.AppendLine("[/SQL]");

            return sb.ToString().Trim();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Console output — replace the body of this method with your LLM call
        // ─────────────────────────────────────────────────────────────────────
        static void EmitChunk(SqlChunk chunk)
        {
            Console.WriteLine(new string('─', 60));
            Console.WriteLine(chunk.FullContextBlock);
            Console.WriteLine();

            // ── PIPELINE NOTE ──────────────────────────────────────────────
            // Pass the following fields to your vector DB / LLM API:
            //
            //   chunk.FullContextBlock      → text to embed / send as context
            //   chunk.ChunkCategory         → filter out "SEED_DATA" when only
            //                                 schema context is needed
            //   chunk.ObjectName            → metadata for faceted retrieval
            //   chunk.ReferencedTables      → expand retrieval to related tables
            //   chunk.FkRelationships       → structured join paths; use to auto-
            //                                 expand context to FK target chunks
            //   chunk.ChunkId == 0          → always inject as system-level context
            // ──────────────────────────────────────────────────────────────
        }
    }
}