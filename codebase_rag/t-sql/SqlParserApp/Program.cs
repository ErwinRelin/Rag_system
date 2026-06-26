using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.SqlServer.TransactSql.ScriptDom;

namespace SqlChunkerApp
{
    class StructuralSignals
    {
        public string ObjectName { get; set; }
        public string ObjectType { get; set; }
        public string Category { get; set; }
        public string RawSql { get; set; }
        public string FileName { get; set; }

        public HashSet<string> WritesTo { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> ReadsFrom { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public int IfStatementCount { get; set; }
        public int ValidationErrorCount { get; set; }
        public bool HasTransactionScope { get; set; }
        public int StateColumnAssignmentCount { get; set; }
        public List<string> Parameters { get; set; } = new();
        public List<string> OutputParameters { get; set; } = new();
        public Dictionary<string, HashSet<string>> ColumnsRead { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> ForeignKeyReferences { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public List<string> StateColumns { get; set; } = new();
        public int DataModificationCount { get; set; }
        public bool HasArchiveOperations { get; set; }
        public bool WritesOnlyToAuditLog { get; set; }
        public int TempTableCount { get; set; }
        public bool HasDynamicSql { get; set; }
        public bool HasSystemMetadataAccess { get; set; }
        public bool HasCursorUsage { get; set; }
        public int CleanupOperationCount { get; set; }
        public bool HasBatchProcessing { get; set; }
        public bool HasWaitForDelay { get; set; }
        public int PrintStatementCount { get; set; }
        public bool HasConfigOnlyParameters { get; set; }
        public int InsertStatementCount { get; set; }
        public int UpdateStatementCount { get; set; }
        public int DeleteStatementCount { get; set; }
        public int MergeStatementCount { get; set; }
        public int SumFunctionCount { get; set; }
        public int CountFunctionCount { get; set; }
        public int WindowFunctionCount { get; set; }
        public bool IsReadOnly { get; set; }
        public int DistinctTableCount { get; set; }
        public int SelectStatementCount { get; set; }
        public int JoinCount { get; set; }
        public int GroupByCount { get; set; }
        public int AggregateFunctionCount { get; set; }
        public int OrderByCount { get; set; }
        public int SubqueryCount { get; set; }
        public int CteCount { get; set; }
        public bool HasUnionOperator { get; set; }
        public bool HasCaseExpression { get; set; }
        public bool HasForXmlPath { get; set; }
        public bool HasPaginationPattern { get; set; }
        public bool HasDateRangeFilter { get; set; }
        public int OutputColumnCount { get; set; }

        public double BusinessLogicScore
        {
            get
            {
                double score = 0.0;
                score += Math.Log(DataModificationCount + 1, 2) * 0.22;
                score += Math.Log(IfStatementCount + 1, 2) * 0.10;
                int clampedVals = Math.Min(ValidationErrorCount, 5);
                score += Math.Log(clampedVals + 1, 2) * 0.12;
                if (HasTransactionScope) score += 0.18;
                score += Math.Log(StateColumnAssignmentCount + 1, 2) * 0.06;
                if (OutputParameters.Count > 0) score += 0.06;
                return Math.Min(1.0, Math.Round(score, 2));
            }
        }

        public double UtilityScore
        {
            get
            {
                double score = 0.0;
                if (HasArchiveOperations) score += 0.30;
                if (WritesOnlyToAuditLog) score += 0.20;
                score += Math.Log(TempTableCount + 1, 2) * 0.12;
                if (HasDynamicSql) score += 0.20;
                if (HasSystemMetadataAccess) score += 0.15;
                if (HasCursorUsage) score += 0.15;
                score += Math.Log(CleanupOperationCount + 1, 2) * 0.10;
                if (HasBatchProcessing) score += 0.15;
                if (HasWaitForDelay) score += 0.15;
                score += Math.Log(PrintStatementCount + 1, 2) * 0.05;
                if (HasConfigOnlyParameters) score += 0.10;
                return Math.Min(1.0, Math.Round(score, 2));
            }
        }

        public double ReportingScore
        {
            get
            {
                double score = 0.0;
                score += Math.Log(GroupByCount + 1, 2) * 0.10;
                score += Math.Log(SumFunctionCount + 1, 2) * 0.08;
                score += Math.Log(CountFunctionCount + 1, 2) * 0.06;
                score += Math.Log(WindowFunctionCount + 1, 2) * 0.12;
                if (IsReadOnly) score += 0.25;
                score += Math.Log(OrderByCount + 1, 2) * 0.05;
                if (HasPaginationPattern) score += 0.10;
                score -= Math.Log(UpdateStatementCount + 1, 2) * 0.15;
                score -= Math.Log(InsertStatementCount + 1, 2) * 0.12;
                score -= Math.Log(DeleteStatementCount + 1, 2) * 0.18;
                score -= Math.Log(MergeStatementCount + 1, 2) * 0.20;
                if (HasTransactionScope) score -= 0.20;
                score -= Math.Log(ValidationErrorCount + 1, 2) * 0.10;
                score -= Math.Log(StateColumnAssignmentCount + 1, 2) * 0.08;
                if (OutputParameters.Count > 0) score -= 0.10;
                if (HasArchiveOperations) score -= 0.25;
                if (HasBatchProcessing) score -= 0.15;
                return Math.Max(-1.0, Math.Min(1.0, Math.Round(score, 2)));
            }
        }

        /// <summary>
        /// Primary intent — what the procedure fundamentally IS.
        /// This IS the chunk classification. Mutually exclusive.
        /// </summary>
        public string ChunkClassification
        {
            get
            {
                // ── ARCHIVE ────────────────────────────────────────────
                if (HasArchiveOperations)
                    return "ARCHIVE";

                // ── REPORT: read-only, returns data ────────────────────
                if (IsReadOnly && (SelectStatementCount > 0 || DistinctTableCount >= 2))
                    return "REPORT";

                // ── PURGE ──────────────────────────────────────────────
                if (CleanupOperationCount > 0 && !HasArchiveOperations 
                    && !HasTransactionScope && DataModificationCount <= 1 
                    && BusinessLogicScore < 0.5
                    && !(IsReadOnly && DistinctTableCount >= 3))  // not a report with incidental cleanup
                    return "PURGE";

                // ── AUDIT ──────────────────────────────────────────────
                if (WritesOnlyToAuditLog)
                    return "AUDIT";

                // ── LOOKUP: read-only, single-table, no aggregates ─────
                if (IsReadOnly && DistinctTableCount == 1 && AggregateFunctionCount == 0)
                    return "LOOKUP";

                // ── CONFIG ─────────────────────────────────────────────
                if (HasConfigOnlyParameters)
                    return "CONFIG";

                // ── BUSINESS_OPERATION ─────────────────────────────────
                if (!IsReadOnly && DataModificationCount > 0)
                    return "BUSINESS_OPERATION";

                return "UTILITY";
            }
        }

        /// <summary>
        /// Secondary traits — capabilities the procedure HAS.
        /// Multiple can apply. Used for filtering and domain attachment.
        /// </summary>
        public List<string> Traits
        {
            get
            {
                var traits = new List<string>();

                // Domain scope
                if (DistinctTableCount >= 4)      traits.Add("CROSS_DOMAIN");
                else if (DistinctTableCount <= 2) traits.Add("SINGLE_DOMAIN");

                // Processing patterns
                if (HasBatchProcessing)       traits.Add("BATCH");
                if (HasTransactionScope)      traits.Add("TRANSACTIONAL");
                if (HasCursorUsage)           traits.Add("CURSOR");
                if (HasWaitForDelay)          traits.Add("SCHEDULED");
                if (TempTableCount > 0)       traits.Add("TEMP_TABLES");

                // Business logic
                if (ValidationErrorCount >= 3) traits.Add("VALIDATED");
                if (StateColumnAssignmentCount >= 2) traits.Add("STATE_MACHINE");
                if (OutputParameters.Count > 0) traits.Add("FACTORY");

                // Query complexity
                if (HasDynamicSql)           traits.Add("DYNAMIC_SQL");
                if (GroupByCount > 0)        traits.Add("AGGREGATED");
                if (HasPaginationPattern)    traits.Add("PAGINATED");
                if (HasForXmlPath)           traits.Add("XML_AGGREGATION");
                if (CteCount > 0)            traits.Add("CTE");
                if (HasSystemMetadataAccess) traits.Add("METADATA");

                return traits;
            }
        }

        public string ChunkDescription
        {
            get
            {
                var sb = new System.Text.StringBuilder();

                sb.Append(ChunkClassification switch
                {
                    "ARCHIVE"            => $"Archive — moves data to history tables",
                    "PURGE"              => $"Purge — deletes old records without archiving",
                    "AUDIT"              => $"Audit — logs to AuditLog without modifying business data",
                    "LOOKUP"             => $"Lookup — simple single-table retrieval",
                    "REPORT"             => $"Report — read-only result set across {DistinctTableCount} tables",
                    "CONFIG"             => $"Configuration — parameter-driven setup",
                    "BUSINESS_OPERATION" => $"Business operation — writes to [{string.Join(", ", WritesTo.Where(t => !t.Equals("AuditLog")))}]",
                    _                    => $"Utility — general-purpose operation"
                });

                if (Traits.Any())
                {
                    sb.Append(". Traits: ");
                    sb.Append(string.Join(", ", Traits.Select(t => t.ToLower().Replace("_", " "))));
                    sb.Append(".");
                }

                return sb.ToString();
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // AST WALKER — single visitor for all ScriptDom signals
    // ═══════════════════════════════════════════════════════════════

    class StructuralSignalAstWalker : TSqlFragmentVisitor
    {
        public int InsertCount { get; private set; }
        public int UpdateCount { get; private set; }
        public int DeleteCount { get; private set; }
        public int IfStatementCount { get; private set; }
        public int ValidationErrorCount { get; private set; }
        public bool HasTransactionScope { get; private set; }
        public HashSet<string> WrittenTables { get; } = new(StringComparer.OrdinalIgnoreCase);
        public List<string> StateColumns { get; } = new();

        private static readonly HashSet<string> StateColumnNames = new(StringComparer.OrdinalIgnoreCase)
        {
            "Status", "IsActive", "IsDeleted", "IsArchived", "State",
            "ModifiedDate", "UpdatedAt", "ExpiryDate", "ClosedDate",
            "FulfilledDate", "PaidDate", "ReturnDate", "WaivedBy", "NotificationSent"
        };

        public override void Visit(InsertStatement node)
        {
            InsertCount++;
            ExtractTarget(node.InsertSpecification?.Target);
            base.Visit(node);
        }

        public override void Visit(UpdateStatement node)
        {
            UpdateCount++;
            ExtractTarget(node.UpdateSpecification?.Target);
            if (node.UpdateSpecification?.SetClauses != null)
            {
                foreach (var clause in node.UpdateSpecification.SetClauses)
                {
                    if (clause is AssignmentSetClause assignment)
                    {
                        string col = assignment.Column?.MultiPartIdentifier?.Identifiers?.LastOrDefault()?.Value;
                        if (!string.IsNullOrEmpty(col) && StateColumnNames.Contains(col))
                            StateColumns.Add(col);
                    }
                }
            }
            base.Visit(node);
        }

        public override void Visit(DeleteStatement node)
        {
            DeleteCount++;
            ExtractTarget(node.DeleteSpecification?.Target);
            base.Visit(node);
        }

        public override void Visit(IfStatement node)
        {
            IfStatementCount++;
            base.Visit(node);
        }

        public override void Visit(RaiseErrorStatement node)
        {
            ValidationErrorCount++;
            base.Visit(node);
        }

        public override void Visit(ThrowStatement node)
        {
            ValidationErrorCount++;
            base.Visit(node);
        }

        public override void Visit(BeginTransactionStatement node)
        {
            HasTransactionScope = true;
            base.Visit(node);
        }

        private void ExtractTarget(TableReference target)
        {
            if (target is NamedTableReference namedTable)
            {
                string name = namedTable.SchemaObject?.BaseIdentifier?.Value;
                if (!string.IsNullOrEmpty(name))
                    WrittenTables.Add(name);
            }
        }
    }

    class ParameterExtractor : TSqlFragmentVisitor
    {
        public List<string> InputParameters { get; } = new();
        public List<string> OutputParameters { get; } = new();

        public override void Visit(CreateProcedureStatement node)
        {
            if (node.Parameters != null)
            {
                foreach (var param in node.Parameters)
                {
                    string name = param.VariableName?.Value ?? "";
                    if (param.Modifier == ParameterModifier.Output || param.Modifier == ParameterModifier.ReadOnly)
                        OutputParameters.Add(name);
                    else
                        InputParameters.Add(name);
                }
            }
            base.Visit(node);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // MAIN PARSER ENGINE
    // ═══════════════════════════════════════════════════════════════

    class StructuralSignalParser
    {
        private readonly TSql160Parser _parser;
        private readonly Sql160ScriptGenerator _generator;

        private static readonly HashSet<string> Keywords = new(StringComparer.OrdinalIgnoreCase)
        {
            "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
            "AS", "ON", "SET", "INTO", "VALUES", "TOP", "DISTINCT", "ORDER", "BY",
            "GROUP", "HAVING", "CASE", "WHEN", "THEN", "ELSE", "END", "BEGIN",
            "TRANSACTION", "COMMIT", "ROLLBACK", "DECLARE", "EXEC", "EXECUTE",
            "RETURN", "IF", "EXISTS", "PRINT", "CAST", "CONVERT", "ISNULL",
            "GETDATE", "DATEADD", "DATEDIFF", "COUNT", "SUM", "AVG", "MAX", "MIN",
            "NOCOUNT", "SCOPE_IDENTITY", "ROWCOUNT", "TRANCOUNT", "FETCH", "OPEN",
            "CLOSE", "DEALLOCATE", "CURSOR", "NEXT", "STATUS", "DAY", "MONTH",
            "YEAR", "LEFT", "RIGHT", "LEN", "SUBSTRING", "CONCAT", "REPLACE",
            "LIKE", "BETWEEN", "INNER", "OUTER", "CROSS", "JOIN", "FULL",
            "WITH", "CTE", "OUTPUT", "INSERTED", "DELETED", "DEFAULT", "IDENTITY",
            "PRIMARY", "FOREIGN", "KEY", "REFERENCES", "CONSTRAINT", "INDEX",
            "CREATE", "ALTER", "DROP", "TABLE", "VIEW", "PROCEDURE", "FUNCTION",
            "TRIGGER", "DATABASE", "USE", "GO", "NVARCHAR", "VARCHAR", "INT",
            "DECIMAL", "BIT", "DATE", "DATETIME", "FLOAT", "NCHAR", "CHAR",
            "BIGINT", "SMALLINT", "TINYINT", "MONEY", "UNIQUEIDENTIFIER",
            "WAITFOR", "DELAY", "WHILE", "BREAK", "CONTINUE", "TRY", "CATCH",
            "RAISERROR", "THROW", "ERROR_MESSAGE", "ERROR_SEVERITY", "ERROR_STATE",
            "ROW_NUMBER", "OVER", "PARTITION", "STUFF", "PATH", "XML", "FOR",
            "SCOPE", "OBJECT", "SYS", "SYSTEM", "STRING_SPLIT", "VALUE",
            "FAST_FORWARD", "LOCAL", "STATIC", "FORWARD_ONLY", "READ_ONLY"
        };

        public StructuralSignalParser()
        {
            _parser = new TSql160Parser(initialQuotedIdentifiers: true);
            _generator = new Sql160ScriptGenerator(new SqlScriptGeneratorOptions
            {
                SqlVersion = SqlVersion.Sql160,
                KeywordCasing = KeywordCasing.Uppercase,
                NewLineBeforeFromClause = true,
                NewLineBeforeJoinClause = true
            });
        }

        public List<StructuralSignals> ParseFile(string filePath)
        {
            string fileName = Path.GetFileName(filePath);
            var results = new List<StructuralSignals>();
            string sqlText = File.ReadAllText(filePath);
            var fragment = _parser.Parse(new StringReader(sqlText), out IList<ParseError> errors);

            if (errors.Count > 0)
            {
                Console.WriteLine($"  Parse errors in {fileName}:");
                foreach (var e in errors) Console.WriteLine($"     Line {e.Line}: {e.Message}");
                return results;
            }

            if (fragment is not TSqlScript script) return results;

            foreach (TSqlBatch batch in script.Batches)
                foreach (TSqlStatement stmt in batch.Statements)
                {
                    var signals = ExtractSignals(stmt, fileName);
                    if (signals != null) results.Add(signals);
                }

            return results;
        }

        private StructuralSignals ExtractSignals(TSqlStatement stmt, string fileName)
        {
            _generator.GenerateScript(stmt, out string rawSql);
            rawSql = rawSql.Trim();
            if (string.IsNullOrWhiteSpace(rawSql)) return null;

            var signals = new StructuralSignals
            {
                ObjectName = ExtractObjectName(stmt),
                ObjectType = stmt.GetType().Name,
                Category = ClassifyStatement(stmt.GetType().Name),
                RawSql = rawSql,
                FileName = fileName
            };

            // ── Single AST walk ────────────────────────────────────
            var walker = new StructuralSignalAstWalker();
            stmt.Accept(walker);

            signals.InsertStatementCount = walker.InsertCount;
            signals.UpdateStatementCount = walker.UpdateCount;
            signals.DeleteStatementCount = walker.DeleteCount;
            signals.HasTransactionScope = walker.HasTransactionScope;

            signals.WritesTo = new HashSet<string>(
                walker.WrittenTables.Where(t => !IsKeyword(t)),
                StringComparer.OrdinalIgnoreCase);

            signals.DataModificationCount = walker.WrittenTables
                .Count(t => !IsKeyword(t) && !t.StartsWith("#")
                    && !t.EndsWith("_Archive")
                    && !t.Equals("AuditLog", StringComparison.OrdinalIgnoreCase)
                    && t.Length > 3);

            signals.StateColumns = walker.StateColumns
                .Distinct(StringComparer.OrdinalIgnoreCase).ToList();
            signals.StateColumnAssignmentCount = signals.StateColumns.Count;

            // ── Regex-based extractions ────────────────────────────
            ExtractReadDependencies(rawSql, signals);
            signals.IfStatementCount = CountBusinessIfStatements(rawSql);
            signals.ValidationErrorCount = CountBusinessValidations(rawSql);
            ExtractUtilitySignals(rawSql, signals);
            ExtractReportingSignals(rawSql, signals);

            signals.WritesOnlyToAuditLog = signals.WritesTo.Count == 1
                && signals.WritesTo.Contains("AuditLog");
            signals.HasConfigOnlyParameters = signals.Parameters.Count > 0
                && signals.Parameters.All(p =>
                    p.Contains("BatchSize") || p.Contains("Retention")
                    || p.Contains("Simulate") || p.Contains("Threshold")
                    || p.Contains("Audit"));
            signals.IsReadOnly = signals.WritesTo.Count == 0
                || (signals.WritesTo.Count == 1 && signals.WritesTo.Contains("AuditLog"));
            signals.DistinctTableCount = signals.ReadsFrom
                .Count(t => !IsKeyword(t) && !t.StartsWith("#") && !t.EndsWith("_cursor"));

            ExtractParameters(stmt, signals);
            ExtractForeignKeyReferences(stmt, signals);

            return signals;
        }

        private void ExtractReadDependencies(string rawSql, StructuralSignals signals)
        {
            var aliasMap = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (Match m in Regex.Matches(rawSql,
                @"\b(?:FROM|JOIN)\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?",
                RegexOptions.IgnoreCase))
            {
                string realTable = m.Groups[1].Value;
                string alias = m.Groups[2].Success ? m.Groups[2].Value : realTable;
                if (!IsKeyword(realTable) && !IsKeyword(alias))
                    aliasMap[alias] = realTable;
            }

            var readPatterns = new[]
            {
                @"\bFROM\s+(\w+)(?:\s+(?:AS\s+)?\w+)?",
                @"\b(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*(?:OUTER\s+)?JOIN\s+(\w+)",
                @"\bEXISTS\s*\(\s*SELECT\s+\d\s+FROM\s+(\w+)"
            };

            foreach (var pattern in readPatterns)
                foreach (Match m in Regex.Matches(rawSql, pattern, RegexOptions.IgnoreCase))
                {
                    string table = aliasMap.TryGetValue(m.Groups[1].Value, out string real) ? real : m.Groups[1].Value;
                    if (!IsKeyword(table)) signals.ReadsFrom.Add(table);
                }

            foreach (Match m in Regex.Matches(rawSql, @"\b(\w+)\.(\w+)\b", RegexOptions.IgnoreCase))
            {
                string aliasOrTable = m.Groups[1].Value, column = m.Groups[2].Value;
                if (IsKeyword(aliasOrTable) || IsKeyword(column)) continue;
                if (Regex.IsMatch(aliasOrTable, @"^\d") || Regex.IsMatch(column, @"^\d")) continue;
                string table = aliasMap.TryGetValue(aliasOrTable, out string resolved) ? resolved : aliasOrTable;
                if (!signals.ColumnsRead.ContainsKey(table))
                    signals.ColumnsRead[table] = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                signals.ColumnsRead[table].Add(column);
            }
        }

        private static int CountBusinessIfStatements(string rawSql)
        {
            var boilerplatePatterns = new[]
            {
                @"IF\s+@@TRANCOUNT\s*>\s*0",
                @"IF\s+@@ERROR\s*<>\s*0",
                @"IF\s+@@FETCH_STATUS",
                @"IF\s+@\w+\s+IS\s+NULL\s*\n\s*BEGIN\s*\n\s*RAISERROR\s*\(\s*'[^']*not\s+found",
            };
            int totalIfs = 0, boilerplateIfs = 0;
            foreach (Match m in Regex.Matches(rawSql, @"\bIF\b", RegexOptions.IgnoreCase))
            {
                totalIfs++;
                string remaining = rawSql.Substring(m.Index, Math.Min(200, rawSql.Length - m.Index));
                foreach (var pattern in boilerplatePatterns)
                    if (Regex.IsMatch(remaining, pattern, RegexOptions.IgnoreCase | RegexOptions.Singleline))
                    { boilerplateIfs++; break; }
            }
            return totalIfs - boilerplateIfs;
        }

        private static int CountBusinessValidations(string rawSql)
        {
            string withoutCatch = Regex.Replace(rawSql, @"BEGIN\s+CATCH.*?END\s+CATCH", "",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);
            return Regex.Matches(withoutCatch, @"\bRAISERROR\b", RegexOptions.IgnoreCase).Count
                 + Regex.Matches(withoutCatch, @"\bTHROW\b", RegexOptions.IgnoreCase).Count;
        }

        private static void ExtractUtilitySignals(string rawSql, StructuralSignals signals)
        {
            signals.HasArchiveOperations = Regex.IsMatch(rawSql,
                @"\b(?:INSERT\s+INTO|DELETE\s+FROM|UPDATE)\s+\w+_Archive\b", RegexOptions.IgnoreCase);
            signals.TempTableCount = Regex.Matches(rawSql, @"\bCREATE\s+TABLE\s+#\w+", RegexOptions.IgnoreCase).Count;
            signals.HasDynamicSql = Regex.IsMatch(rawSql, @"\b(?:EXEC\s*\(\s*@|sp_executesql)", RegexOptions.IgnoreCase);
            signals.HasSystemMetadataAccess = Regex.IsMatch(rawSql,
                @"\b(?:sys\.(?:tables|columns|objects|schemas|databases|indexes|views|procedures)|INFORMATION_SCHEMA\.)", RegexOptions.IgnoreCase);
            signals.HasCursorUsage = Regex.IsMatch(rawSql, @"\bDECLARE\s+\w+\s+CURSOR\b", RegexOptions.IgnoreCase);
            signals.CleanupOperationCount = Regex.Matches(rawSql,
                @"\b(?:DELETE\s+(?:TOP\s*\(.*?\)\s*)?(?:FROM\s+)?\w+|DROP\s+TABLE\s+|TRUNCATE\s+TABLE\s+)", RegexOptions.IgnoreCase).Count;
            signals.HasBatchProcessing = Regex.IsMatch(rawSql,
                @"\b(?:TOP\s*\(\s*@BatchSize\s*\)|@@ROWCOUNT|WHILE\s+@RowCount\s*[<>=])", RegexOptions.IgnoreCase);
            signals.HasWaitForDelay = Regex.IsMatch(rawSql, @"\bWAITFOR\s+DELAY\b", RegexOptions.IgnoreCase);
            signals.PrintStatementCount = Regex.Matches(rawSql, @"\bPRINT\s+", RegexOptions.IgnoreCase).Count;
        }

        private static void ExtractReportingSignals(string rawSql, StructuralSignals signals)
        {
            string cleaned = Regex.Replace(rawSql, @"INSERT\s+INTO.*?SELECT\s+", "",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);

            signals.SelectStatementCount = Regex.Matches(cleaned,
                @"\bSELECT\s+(?!@)\w+",
                RegexOptions.IgnoreCase).Count;
            signals.JoinCount = Regex.Matches(cleaned,
                @"\b(?:INNER|LEFT|RIGHT|FULL|CROSS)?\s*(?:OUTER\s+)?JOIN\b", RegexOptions.IgnoreCase).Count;
            signals.GroupByCount = Regex.Matches(cleaned, @"\bGROUP\s+BY\b", RegexOptions.IgnoreCase).Count;
            signals.AggregateFunctionCount = Regex.Matches(cleaned,
                @"\b(?:COUNT|SUM|AVG|MIN|MAX)\s*\(", RegexOptions.IgnoreCase).Count;
            signals.OrderByCount = Regex.Matches(cleaned, @"\bORDER\s+BY\b", RegexOptions.IgnoreCase).Count;
            signals.SubqueryCount = Regex.Matches(cleaned, @"\(\s*SELECT\s+(?!1\b)\w+", RegexOptions.IgnoreCase).Count;
            signals.CteCount = Regex.Matches(rawSql, @"\bWITH\s+\w+\s+AS\s*\(", RegexOptions.IgnoreCase).Count;
            signals.HasUnionOperator = Regex.IsMatch(rawSql, @"\bUNION\s+(?:ALL\s+)?\b", RegexOptions.IgnoreCase);
            signals.HasCaseExpression = Regex.IsMatch(rawSql, @"\bCASE\s+(?:WHEN|@)", RegexOptions.IgnoreCase);
            signals.HasForXmlPath = Regex.IsMatch(rawSql, @"\bFOR\s+XML\s+PATH\b", RegexOptions.IgnoreCase);
            signals.HasPaginationPattern = Regex.IsMatch(rawSql,
                @"\b(?:TOP\s*\(|ROW_NUMBER\s*\(\s*\)\s+OVER|OFFSET\s+\d+\s+ROWS)", RegexOptions.IgnoreCase);
            signals.HasDateRangeFilter = Regex.IsMatch(rawSql,
                @"\b(?:BETWEEN\s+@\w+\s+AND\s+@\w+|DATEADD\s*\(.*GETDATE)", RegexOptions.IgnoreCase);
            signals.SumFunctionCount = Regex.Matches(rawSql, @"\bSUM\s*\(", RegexOptions.IgnoreCase).Count;
            signals.CountFunctionCount = Regex.Matches(rawSql, @"\bCOUNT\s*\(", RegexOptions.IgnoreCase).Count;
            signals.WindowFunctionCount = Regex.Matches(rawSql,
                @"\b(?:ROW_NUMBER|RANK|DENSE_RANK|NTILE|LAG|LEAD|FIRST_VALUE|LAST_VALUE)\s*\(", RegexOptions.IgnoreCase).Count;
            signals.MergeStatementCount = Regex.Matches(rawSql, @"\bMERGE\s+", RegexOptions.IgnoreCase).Count;

            var selectMatches = Regex.Matches(cleaned, @"\bSELECT\s+(?!@)(?!1\b)(.*?)\bFROM\b",
                RegexOptions.IgnoreCase | RegexOptions.Singleline);
            int totalCols = 0;
            foreach (Match sm in selectMatches)
                totalCols += sm.Groups[1].Value.Count(c => c == ',') + 1;
            signals.OutputColumnCount = totalCols;
        }

        private void ExtractParameters(TSqlStatement stmt, StructuralSignals signals)
        {
            var extractor = new ParameterExtractor();
            stmt.Accept(extractor);
            signals.Parameters = extractor.InputParameters;
            signals.OutputParameters = extractor.OutputParameters;
        }

        private void ExtractForeignKeyReferences(TSqlStatement stmt, StructuralSignals signals)
        {
            if (stmt is not CreateTableStatement createTable) return;
            foreach (var constraint in createTable.Definition.TableConstraints)
                if (constraint is ForeignKeyConstraintDefinition fk)
                { var t = fk.ReferenceTableName?.BaseIdentifier?.Value; if (!string.IsNullOrEmpty(t)) { signals.ForeignKeyReferences.Add(t); signals.ReadsFrom.Add(t); } }
            foreach (var col in createTable.Definition.ColumnDefinitions)
                foreach (var constraint in col.Constraints)
                    if (constraint is ForeignKeyConstraintDefinition fk)
                    { var t = fk.ReferenceTableName?.BaseIdentifier?.Value; if (!string.IsNullOrEmpty(t)) { signals.ForeignKeyReferences.Add(t); signals.ReadsFrom.Add(t); } }
        }

        private static string ClassifyStatement(string objectType) => objectType switch
        {
            "CreateTableStatement" => "DDL", "CreateDatabaseStatement" => "DDL",
            "UseStatement" => "DDL", "AlterTableStatement" => "DDL",
            "CreateIndexStatement" => "DDL", "CreateViewStatement" => "VIEW",
            "CreateProcedureStatement" => "PROCEDURE", "CreateFunctionStatement" => "PROCEDURE",
            "InsertStatement" => "SEED_DATA", "UpdateStatement" => "DML",
            "DeleteStatement" => "DML", "SelectStatement" => "DML", _ => "OTHER"
        };

        private static string ExtractObjectName(TSqlStatement stmt) => stmt switch
        {
            CreateTableStatement s => s.SchemaObjectName?.BaseIdentifier?.Value ?? "",
            CreateViewStatement s => s.SchemaObjectName?.BaseIdentifier?.Value ?? "",
            CreateProcedureStatement s => s.ProcedureReference?.Name?.BaseIdentifier?.Value ?? "",
            CreateFunctionStatement s => s.Name?.BaseIdentifier?.Value ?? "",
            CreateDatabaseStatement s => s.DatabaseName?.Value ?? "", _ => ""
        };

        private static bool IsKeyword(string token) =>
            Keywords.Contains(token) || Regex.IsMatch(token, @"^\d")
            || token.StartsWith("#") || token.EndsWith("_cursor");
    }

    class BusinessDomain
    {
        public string DomainName { get; set; }
        public string BusinessFlow { get; set; }
        public HashSet<string> OwnedTables { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public List<StructuralSignals> Procedures { get; set; } = new();
        public HashSet<string> AllReadTables { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public HashSet<string> AllWriteTables { get; set; } = new(StringComparer.OrdinalIgnoreCase);
    }

    class BusinessCapabilityDiscoverer
    {
        // ── Business capability taxonomy ──────────────────────────
        private static readonly Dictionary<string, string[]> CapabilityVerbs = new(StringComparer.OrdinalIgnoreCase)
        {
            ["MemberManagement"] = new[] { "Member", "Membership" },
            ["Lending"]          = new[] { "Loan", "Issue", "Return", "Renew", "Reserve", "Borrow" },
            ["FineManagement"]   = new[] { "Fine", "Pay", "Waive", "Penalty", "Overdue" },
            ["CatalogManagement"] = new[] { "Book", "Author", "Publisher", "Category", "Catalog", "ISBN", "Title" },
            ["EmployeeManagement"] = new[] { "Employee", "Hire", "Staff", "Terminate" },
            ["LeaveManagement"]  = new[] { "Leave", "Vacation", "Sick", "TimeOff" },
            ["AttendanceTracking"] = new[] { "Attendance", "TimeIn", "TimeOut", "Clock" },
            ["PayrollProcessing"] = new[] { "Payroll", "Salary", "Pay", "Compensation" },
            ["PerformanceManagement"] = new[] { "Performance", "Review", "Appraisal", "Rating" },
            ["TrainingDevelopment"] = new[] { "Training", "Enroll", "Course", "Certification" },
            ["RecruitmentHiring"] = new[] { "Recruit", "Candidate", "Requisition", "Hire", "Job" },
        };

        // ── Entity lifecycle stages ───────────────────────────────
        private static readonly Dictionary<string, string[]> LifecycleStages = new(StringComparer.OrdinalIgnoreCase)
        {
            ["Member"]   = new[] { "Add", "Register", "Activate", "Renew", "Expire", "Deactivate", "Terminate" },
            ["Loan"]     = new[] { "Issue", "Checkout", "Renew", "Return", "Close", "Overdue" },
            ["Fine"]     = new[] { "Assess", "Pay", "Waive", "Appeal", "WriteOff" },
            ["Employee"] = new[] { "Add", "Hire", "Onboard", "Promote", "Transfer", "Terminate", "Exit" },
            ["Leave"]    = new[] { "Submit", "Approve", "Reject", "Cancel", "Take" },
            ["Payroll"]  = new[] { "Calculate", "Process", "Approve", "Distribute", "Close" },
        };

        /// <summary>
        /// Discovers business capabilities from procedure names, state patterns,
        /// and workflow chains — not from table co-occurrence.
        /// </summary>
        public Dictionary<string, BusinessCapability> DiscoverCapabilities(List<StructuralSignals> procs)
        {
            var capabilities = new Dictionary<string, BusinessCapability>(StringComparer.OrdinalIgnoreCase);

            foreach (var proc in procs)
            {
                // Step 1: Extract business nouns and verbs from procedure name
                var (nouns, verbs) = ParseProcedureName(proc.ObjectName);

                // Step 2: Match to known business capabilities
                string capability = MatchCapability(proc, nouns, verbs);

                // Step 3: Determine lifecycle stage
                string stage = MatchLifecycleStage(proc, nouns, verbs);

                // Step 4: Find workflow neighbors (procs that share reads/writes)
                var neighbors = FindWorkflowNeighbors(proc, procs);

                if (!capabilities.ContainsKey(capability))
                {
                    capabilities[capability] = new BusinessCapability
                    {
                        CapabilityName = capability,
                        LifecycleStages = new List<string>(),
                        Procedures = new List<StructuralSignals>(),
                        CoreEntities = new HashSet<string>(StringComparer.OrdinalIgnoreCase),
                        WorkflowEdges = new List<(string from, string to)>(),
                    };
                }

                var cap = capabilities[capability];
                cap.Procedures.Add(proc);
                cap.CoreEntities.UnionWith(nouns);
                if (!cap.LifecycleStages.Contains(stage))
                    cap.LifecycleStages.Add(stage);

                foreach (var neighbor in neighbors)
                {
                    cap.WorkflowEdges.Add((proc.ObjectName, neighbor.ObjectName));
                }
            }

            // Step 5: Order lifecycle stages
            foreach (var cap in capabilities.Values)
            {
                cap.LifecycleOrder = OrderLifecycleStages(cap.LifecycleStages);
            }

            return capabilities;
        }

        private (List<string> nouns, List<string> verbs) ParseProcedureName(string procName)
        {
            // sp_AddNewEmployee → nouns: [Employee], verbs: [Add]
            // sp_SubmitLeaveRequest → nouns: [Leave, Request], verbs: [Submit]
            // sp_ProcessOverdueBooks → nouns: [Book], verbs: [Process]
            
            string name = procName.Replace("sp_", "").Replace("_", "");
            
            var nouns = new List<string>();
            var verbs = new List<string>();

            // Known verbs
            foreach (var verb in new[] { "Add", "Create", "Submit", "Approve", "Reject", 
                "Process", "Generate", "Get", "Search", "Update", "Delete", "Terminate",
                "Issue", "Return", "Renew", "Reserve", "Pay", "Waive", "Transfer",
                "Record", "Enroll", "Bulk", "Calculate" })
            {
                if (name.StartsWith(verb, StringComparison.OrdinalIgnoreCase))
                {
                    verbs.Add(verb);
                    name = name.Substring(verb.Length);
                    break;
                }
            }

            // Known entity nouns
            foreach (var noun in new[] { "Employee", "Member", "Book", "Loan", "Fine", 
                "Payment", "Leave", "Request", "Attendance", "Payroll", "Performance",
                "Review", "Training", "Candidate", "Recruitment", "Reservation",
                "Membership", "Author", "Publisher", "Category", "Department",
                "Position", "Staff", "Dashboard", "Report", "Statement", "History",
                "Inventory", "Audit", "Record" })
            {
                if (name.Contains(noun, StringComparison.OrdinalIgnoreCase))
                {
                    nouns.Add(noun);
                }
            }

            return (nouns, verbs);
        }

        private string MatchCapability(StructuralSignals proc, List<string> nouns, List<string> verbs)
        {
            // Try noun matching first (most specific)
            foreach (var noun in nouns)
            {
                foreach (var (capability, keywords) in CapabilityVerbs)
                {
                    if (keywords.Any(k => k.Equals(noun, StringComparison.OrdinalIgnoreCase) ||
                                        noun.Contains(k, StringComparison.OrdinalIgnoreCase)))
                    {
                        return capability;
                    }
                }
            }

            // Fall back to verb matching
            foreach (var verb in verbs)
            {
                foreach (var (capability, keywords) in CapabilityVerbs)
                {
                    if (keywords.Any(k => k.Equals(verb, StringComparison.OrdinalIgnoreCase)))
                    {
                        return capability;
                    }
                }
            }

            // Fall back to write-target-based naming
            string primaryTable = proc.WritesTo
                .FirstOrDefault(t => !t.Equals("AuditLog") && !t.StartsWith("#") && t.Length > 3);
            
            return primaryTable != null ? $"{primaryTable}Management" : "GeneralOperations";
        }

        private string MatchLifecycleStage(StructuralSignals proc, List<string> nouns, List<string> verbs)
        {
            foreach (var noun in nouns)
            {
                if (LifecycleStages.TryGetValue(noun, out var stages))
                {
                    foreach (var verb in verbs)
                    {
                        var match = stages.FirstOrDefault(s => 
                            s.Equals(verb, StringComparison.OrdinalIgnoreCase));
                        if (match != null) return match;
                    }
                }
            }
            
            return verbs.FirstOrDefault() ?? "Process";
        }

        private List<StructuralSignals> FindWorkflowNeighbors(StructuralSignals proc, List<StructuralSignals> allProcs)
        {
            return allProcs
                .Where(other => other != proc)
                .Where(other =>
                {
                    // Share write targets (same table, different operation)
                    bool sharedWrites = proc.WritesTo
                        .Intersect(other.WritesTo, StringComparer.OrdinalIgnoreCase)
                        .Any(t => !t.Equals("AuditLog") && !t.StartsWith("#"));

                    // Share read targets with complementary verbs
                    bool sharedReads = proc.ReadsFrom
                        .Intersect(other.ReadsFrom, StringComparer.OrdinalIgnoreCase)
                        .Count() >= 2;

                    // One writes what the other reads (producer-consumer)
                    bool producerConsumer = proc.WritesTo
                        .Intersect(other.ReadsFrom, StringComparer.OrdinalIgnoreCase)
                        .Any(t => !t.Equals("AuditLog"));

                    return sharedWrites || sharedReads || producerConsumer;
                })
                .Take(5) // limit workflow neighbors
                .ToList();
        }

        private string OrderLifecycleStages(List<string> stages)
        {
            var orderedStages = new[] { 
                "Add", "Register", "Create", "Submit", "Record", "Enroll",
                "Issue", "Checkout", "Reserve",
                "Approve", "Process", "Calculate", "Generate",
                "Renew", "Update", "Transfer", "Promote",
                "Return", "Close", "Pay", "Waive",
                "Expire", "Deactivate", "Terminate", "Exit", "Cancel", "Reject",
                "Delete", "Purge", "Archive"
            };

            return string.Join(" → ", stages
                .OrderBy(s => Array.IndexOf(orderedStages, s))
                .Distinct());
        }
    }

    class BusinessCapability
    {
        public string CapabilityName { get; set; }
        public List<string> LifecycleStages { get; set; }
        public string LifecycleOrder { get; set; }
        public List<StructuralSignals> Procedures { get; set; }
        public HashSet<string> CoreEntities { get; set; }
        public List<(string from, string to)> WorkflowEdges { get; set; }
    }

    static class SignalDiagnostics
    {
        public static void PrintAll(List<StructuralSignals> signals)
        {
            Console.WriteLine("\n═══════════════════════════════════════════");
            Console.WriteLine("  STRUCTURAL SIGNALS EXTRACTED");
            Console.WriteLine("═══════════════════════════════════════════\n");
            var procs = signals.Where(s => s.Category == "PROCEDURE").ToList();
            foreach (var s in procs)
            {
                Console.WriteLine($"┌─ {s.ObjectName}");
                Console.WriteLine($"├─ WritesTo: [{string.Join(", ", s.WritesTo)}]");
                Console.WriteLine($"├─ ReadsFrom: [{string.Join(", ", s.ReadsFrom)}]");
                Console.WriteLine($"├─ Classification: {s.ChunkClassification}");
                Console.WriteLine($"├─ Traits: [{(s.Traits.Any() ? string.Join(", ", s.Traits) : "none")}]");
                Console.WriteLine($"├─ Business: {s.BusinessLogicScore:F2} | Utility: {s.UtilityScore:F2} | Reporting: {s.ReportingScore:F2}");
                Console.WriteLine($"├─ DML: INSERT={s.InsertStatementCount} UPDATE={s.UpdateStatementCount} DELETE={s.DeleteStatementCount}");
                Console.WriteLine($"├─ IFs={s.IfStatementCount} Vals={s.ValidationErrorCount} Txn={s.HasTransactionScope} States={s.StateColumnAssignmentCount}");
                Console.WriteLine($"├─ {s.ChunkDescription}");
                Console.WriteLine();
            }
            Console.WriteLine($"Total: {signals.Count} | Procs: {procs.Count} | DDL: {signals.Count(s => s.Category == "DDL")} | Seed: {signals.Count(s => s.Category == "SEED_DATA")}");
        }
    }

    class Program
    {
        static void PrintDomainClusters(List<BusinessDomain> domains)
        {
            Console.WriteLine("\n═══════════════════════════════════════════");
            Console.WriteLine("  BUSINESS CAPABILITIES");
            Console.WriteLine("═══════════════════════════════════════════\n");

            foreach (var domain in domains)
            {
                Console.WriteLine($"┌─ {domain.DomainName}");
                Console.WriteLine($"├─ Lifecycle: {domain.BusinessFlow}");
                Console.WriteLine($"├─ Core Entities: [{(domain.OwnedTables.Any() ? string.Join(", ", domain.OwnedTables) : "none")}]");
                Console.WriteLine($"├─ Procedures ({domain.Procedures.Count}):");
                
                foreach (var proc in domain.Procedures.OrderBy(p => GetStageOrder(p, domain.BusinessFlow)))
                {
                    string classification = proc.ChunkClassification;
                    string stage = GetLifecycleStage(proc);
                    string traits = proc.Traits.Any() 
                        ? $" [{string.Join(", ", proc.Traits)}]" 
                        : "";
                    
                    Console.WriteLine($"│  ├─ {stage,-12} {proc.ObjectName,-35} ({classification}){traits}");
                }
                Console.WriteLine("│");
                Console.WriteLine();
            }

            // ── Summary table ──────────────────────────────────────────
            Console.WriteLine("───────────────────────────────────────────");
            Console.WriteLine("  CAPABILITY SUMMARY");
            Console.WriteLine("───────────────────────────────────────────");
            Console.WriteLine($"  {"Capability",-30} {"Procs",-8} {"Entities",-12} {"Lifecycle"}");
            Console.WriteLine($"  {"──────────",-30} {"─────",-8} {"────────",-12} {"─────────"}");
            
            foreach (var domain in domains)
            {
                Console.WriteLine($"  {domain.DomainName,-30} {domain.Procedures.Count,-8} " +
                    $"{domain.OwnedTables.Count,-12} {domain.BusinessFlow}");
            }
            Console.WriteLine();
            Console.WriteLine($"  Total capabilities: {domains.Count}");
            Console.WriteLine($"  Total procedures: {domains.Sum(d => d.Procedures.Count)}");
            
            // ── Cross-capability dependencies ──────────────────────────
            Console.WriteLine();
            Console.WriteLine("───────────────────────────────────────────");
            Console.WriteLine("  CROSS-CAPABILITY REFERENCES");
            Console.WriteLine("───────────────────────────────────────────");
            foreach (var domain in domains)
            {
                var externalReads = domain.Procedures
                    .SelectMany(p => p.ReadsFrom)
                    .Where(t => !domain.OwnedTables.Contains(t))
                    .Where(t => !t.Equals("AuditLog") && !t.StartsWith("#") && t.Length > 3)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();
                
                if (externalReads.Any())
                {
                    Console.WriteLine($"  {domain.DomainName} reads from: [{string.Join(", ", externalReads)}]");
                }
            }
        }

        private static string GetLifecycleStage(StructuralSignals proc)
        {
            // Extract verb from procedure name
            string name = proc.ObjectName.Replace("sp_", "");
            foreach (var verb in new[] { "Add", "Create", "Submit", "Approve", "Reject",
                "Process", "Generate", "Get", "Search", "Update", "Delete", "Terminate",
                "Issue", "Return", "Renew", "Reserve", "Pay", "Waive", "Transfer",
                "Record", "Enroll", "Bulk", "Calculate" })
            {
                if (name.StartsWith(verb, StringComparison.OrdinalIgnoreCase))
                    return verb;
            }
            return "Execute";
        }

        private static int GetStageOrder(StructuralSignals proc, string lifecycle)
        {
            var stages = lifecycle.Split(" → ");
            string stage = GetLifecycleStage(proc);
            for (int i = 0; i < stages.Length; i++)
                if (stages[i].Equals(stage, StringComparison.OrdinalIgnoreCase))
                    return i;
            return 999;
        }

        static void Main(string[] args)
        {
            string targetFolder = @"C:\Users\Erwin\Desktop\rag_system\sql_files";
            string signalsPath = @"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql\shared_data\structural_signals.json";
            string domainsPath = @"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql\shared_data\domains.json";

            if (!Directory.Exists(targetFolder)) 
            { 
                Console.WriteLine($"Folder not found: {targetFolder}"); 
                return; 
            }
            Directory.CreateDirectory(Path.GetDirectoryName(signalsPath));

            // ── Parse all SQL files ───────────────────────────────
            var parser = new StructuralSignalParser();
            var allSignals = new List<StructuralSignals>();
            
            foreach (string filePath in Directory.GetFiles(targetFolder, "*.sql"))
            {
                Console.WriteLine($"Processing: {Path.GetFileName(filePath)}");
                allSignals.AddRange(parser.ParseFile(filePath));
            }

            // ── Print classified signals ──────────────────────────
            SignalDiagnostics.PrintAll(allSignals);

            // ── Save signals to JSON ──────────────────────────────
            var signalsJson = JsonSerializer.Serialize(allSignals,
                new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
            File.WriteAllText(signalsPath, signalsJson);
            Console.WriteLine($"Signals saved → {signalsPath}");

            // ── Cluster into domains ──────────────────────────────
            // ── Discover business capabilities ────────────────────
            var capabilityDiscoverer = new BusinessCapabilityDiscoverer();
            var capabilities = capabilityDiscoverer.DiscoverCapabilities(
                allSignals.Where(s => s.Category == "PROCEDURE").ToList());

            // ── Convert capabilities to domains ───────────────────
            var domains = capabilities.Select(cap => new BusinessDomain
            {
                DomainName = cap.Value.CapabilityName,
                BusinessFlow = cap.Value.LifecycleOrder,
                OwnedTables = new HashSet<string>(cap.Value.CoreEntities, StringComparer.OrdinalIgnoreCase),
                Procedures = cap.Value.Procedures,
                AllReadTables = new HashSet<string>(
                    cap.Value.Procedures.SelectMany(p => p.ReadsFrom), 
                    StringComparer.OrdinalIgnoreCase),
                AllWriteTables = new HashSet<string>(
                    cap.Value.Procedures.SelectMany(p => p.WritesTo), 
                    StringComparer.OrdinalIgnoreCase)
            }).ToList();

            // ── Print domain clusters ─────────────────────────────
            PrintDomainClusters(domains);

            // ── Save domains to JSON ──────────────────────────────
            var domainsJson = JsonSerializer.Serialize(domains,
                new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
            File.WriteAllText(domainsPath, domainsJson);
            Console.WriteLine($"Domains saved → {domainsPath}");

            Console.WriteLine($"\nDone! {allSignals.Count} signals → {domains.Count} domains");
        }
    }
}