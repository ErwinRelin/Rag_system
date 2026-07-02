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
        /// Renders a SemanticIntent into natural English.
        /// No raw counts appear in the output.
        /// </summary>
        static class SemanticSummaryRenderer
        {
            public static string Render(SemanticIntent intent)
            {
                var sb = new System.Text.StringBuilder();

                // ── Core operation ─────────────────────────────────────
                sb.Append($"{intent.Operation} {intent.PrimaryEntity}");

                // ── Related entities ───────────────────────────────────
                if (intent.RelatedEntities.Any())
                {
                    sb.Append($" involving {string.Join(", ", intent.RelatedEntities)}");
                }

                sb.Append(". ");

                // ── Business rules ─────────────────────────────────────
                if (intent.BusinessRules.Any())
                {
                    sb.Append("Enforces ");
                    sb.Append(string.Join(" with ", intent.BusinessRules));
                    sb.Append(". ");
                }

                // ── State transitions ──────────────────────────────────
                if (intent.StateTransitions.Any())
                {
                    sb.Append(string.Join(". ", intent.StateTransitions.Select(t => char.ToUpper(t[0]) + t.Substring(1))));
                    sb.Append(". ");
                }

                // ── Outputs ────────────────────────────────────────────
                if (intent.Outputs.Any())
                {
                    sb.Append("Upon completion, ");
                    sb.Append(string.Join(" and ", intent.Outputs));
                    sb.Append(". ");
                }

                // ── Side effects ───────────────────────────────────────
                if (intent.SideEffects.Any())
                {
                    sb.Append("Additionally, ");
                    sb.Append(string.Join(", ", intent.SideEffects));
                    sb.Append(". ");
                }

                // ── Behavioral context ─────────────────────────────────
                sb.Append($"Operates as a{("aeiou".Contains(intent.ProcessingStyle[0]) ? "n " : " ")}");
                sb.Append($"{intent.ProcessingStyle} process at {intent.DataScope} scope");

                if (intent.IsAtomic)
                    sb.Append(" with transactional integrity");

                sb.Append(".");

                return sb.ToString();
            }
        }

        // Remove the old SemanticSummary property, replace with:
        public string SemanticSummary
        {
            get
            {
                var intent = SemanticIntentExtractor.Extract(this);
                return SemanticSummaryRenderer.Render(intent);
            }
        }

        private string DetermineOperationType()
        {
            if (OutputParameters.Count > 0 && InsertStatementCount > 0)
                return "entity creation";
            if (StateColumnAssignmentCount >= 2)
                return "lifecycle state transition";
            if (UpdateStatementCount > InsertStatementCount)
                return "entity update";
            if (DeleteStatementCount > 0)
                return "entity deletion";
            if (InsertStatementCount > 0)
                return "data insertion";
            return "data modification";
        }

        private List<string> ExtractBusinessEntities()
        {
            var entities = new List<string>();
            string name = ObjectName.Replace("sp_", "").Replace("_", "");
            
            // Extract nouns from procedure name using camelCase splitting
            var words = new List<string>();
            int start = 0;
            for (int i = 1; i < name.Length; i++)
                if (char.IsUpper(name[i]) && !char.IsUpper(name[i - 1]))
                { words.Add(name.Substring(start, i - start)); start = i; }
            if (start < name.Length) words.Add(name.Substring(start));

            // First word is usually a verb, skip it
            foreach (var word in words.Skip(1))
                if (word.Length > 2 && !IsCommonWord(word))
                    entities.Add(word);

            // If no entities found in name, use primary write target
            if (!entities.Any())
            {
                var primaryWrite = WritesTo.FirstOrDefault(t => 
                    !t.Equals("AuditLog") && !t.StartsWith("#") && t.Length > 3);
                if (primaryWrite != null)
                    entities.Add(primaryWrite);
            }

            return entities;
        }

        private bool IsCommonWord(string word)
        {
            var common = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            {
                "New", "Old", "All", "By", "Top", "Get", "Set", "Bulk", "Monthly",
                "Daily", "Annual", "Current", "Next", "Previous"
            };
            return common.Contains(word);
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
                if ((IsReadOnly || WritesOnlyToAuditLog) && (SelectStatementCount > 0 || DistinctTableCount >= 2))
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
                sb.AppendLine($"[{ChunkClassification}] {SemanticSummary}");
                if (Traits.Any())
                    sb.Append($" Traits: {string.Join(", ", Traits.Select(t => t.ToLower().Replace("_", " ")))}.");
                return sb.ToString().Trim();
            }
        }
    }

    // DELETE everything from "class SemanticIntent" inside StructuralSignals 
    // down to the closing brace of RagChunkEmitter

    // ADD these at namespace level, after StructuralSignals closing brace:

    class SemanticIntent
    {
        public string Operation { get; set; }
        public string PrimaryEntity { get; set; }
        public List<string> RelatedEntities { get; set; } = new();
        public List<string> BusinessRules { get; set; } = new();
        public List<string> StateTransitions { get; set; } = new();
        public List<string> Outputs { get; set; } = new();
        public List<string> SideEffects { get; set; } = new();
        public bool IsAtomic { get; set; }
        public bool SpansMultipleDomains { get; set; }
        public string ProcessingStyle { get; set; }
        public string DataScope { get; set; }
    }

    static class SemanticIntentExtractor
    {
        public static SemanticIntent Extract(StructuralSignals s)
        {
            var intent = new SemanticIntent();
            intent.Operation = DetermineOperation(s);
            intent.PrimaryEntity = ExtractPrimaryEntity(s);
            intent.RelatedEntities = ExtractRelatedEntities(s);
            intent.BusinessRules = ExtractBusinessRules(s);
            intent.StateTransitions = ExtractStateTransitions(s);
            intent.Outputs = ExtractOutputs(s);
            intent.SideEffects = ExtractSideEffects(s);
            intent.IsAtomic = s.HasTransactionScope;
            intent.SpansMultipleDomains = s.DistinctTableCount >= 4;
            intent.ProcessingStyle = DetermineProcessingStyle(s);
            intent.DataScope = DetermineDataScope(s);
            return intent;
        }

        private static string DetermineOperation(StructuralSignals s) => s.ChunkClassification switch
        {
            "BUSINESS_OPERATION" when s.OutputParameters.Count > 0 && s.InsertStatementCount > 0 => "Creates",
            "BUSINESS_OPERATION" when s.StateColumnAssignmentCount >= 2 => "Progresses",
            "BUSINESS_OPERATION" when s.DeleteStatementCount > 0 => "Removes",
            "BUSINESS_OPERATION" when s.UpdateStatementCount > s.InsertStatementCount => "Updates",
            "BUSINESS_OPERATION" => "Modifies",
            "REPORT" when s.GroupByCount > 0 => "Summarizes",
            "REPORT" => "Reports on",
            "ARCHIVE" => "Archives",
            "PURGE" => "Purges",
            "AUDIT" => "Audits",
            "LOOKUP" => "Retrieves",
            "CONFIG" => "Configures",
            _ => "Processes"
        };

        private static string ExtractPrimaryEntity(StructuralSignals s)
        {
            string name = s.ObjectName.Replace("sp_", "").Replace("_", "");
            var words = SplitCamelCase(name);
            var entities = words.Skip(1).Where(w => w.Length > 2 && !IsStopWord(w)).ToList();
            if (entities.Any()) return string.Join(" ", entities).ToLower();
            var pw = s.WritesTo.FirstOrDefault(t => !IsInfrastructure(t));
            return pw?.ToLower() ?? "data";
        }

        private static List<string> ExtractRelatedEntities(StructuralSignals s)
        {
            var entities = new List<string>();
            entities.AddRange(s.ReadsFrom.Where(t => !s.WritesTo.Contains(t) && !IsInfrastructure(t)).Take(3).Select(HumanizeTableName));
            var pw = s.WritesTo.FirstOrDefault(t => !IsInfrastructure(t));
            entities.AddRange(s.WritesTo.Where(t => !IsInfrastructure(t) && !t.Equals(pw)).Take(2).Select(HumanizeTableName));
            return entities.Distinct().ToList();
        }

        private static List<string> ExtractBusinessRules(StructuralSignals s)
        {
            var rules = new List<string>();
            if (s.ValidationErrorCount >= 3) rules.Add("multiple validation checks");
            else if (s.ValidationErrorCount > 0) rules.Add("input validation");
            if (s.IfStatementCount >= 5) rules.Add("complex business logic branching");
            else if (s.IfStatementCount >= 2) rules.Add("conditional business rules");
            if (s.HasBatchProcessing) rules.Add("batch processing safeguards");
            if (s.OutputParameters.Count > 0) rules.Add("entity creation with identity return");
            return rules;
        }

        private static List<string> ExtractStateTransitions(StructuralSignals s)
        {
            var transitions = new List<string>();
            var cols = s.StateColumns.Distinct().ToList();
            if (cols.Contains("Status")) transitions.Add("manages record status lifecycle");
            if (cols.Contains("IsActive")) transitions.Add("handles activation and deactivation");
            if (cols.Contains("ExpiryDate") || cols.Contains("ClosedDate")) transitions.Add("manages expiration timeline");
            if (cols.Contains("PaidDate") || cols.Contains("ReturnDate")) transitions.Add("tracks completion dates");
            if (!transitions.Any() && s.StateColumnAssignmentCount >= 2) transitions.Add("manages business state transitions");
            return transitions;
        }

        private static List<string> ExtractOutputs(StructuralSignals s)
        {
            var outputs = new List<string>();
            if (s.OutputParameters.Count > 0) outputs.Add($"returns new {ExtractPrimaryEntity(s)} identifier");
            if (s.SelectStatementCount > 0 && !s.IsReadOnly) outputs.Add("returns confirmation data");
            if (s.SelectStatementCount >= 3 && s.IsReadOnly) outputs.Add("produces multi-section report");
            else if (s.SelectStatementCount > 0 && s.IsReadOnly) outputs.Add("produces result set");
            if (s.GroupByCount > 0) outputs.Add($"with {s.GroupByCount} levels of aggregation");
            return outputs;
        }

        private static List<string> ExtractSideEffects(StructuralSignals s)
        {
            var effects = new List<string>();
            if (s.WritesTo.Contains("AuditLog")) effects.Add("maintains audit trail");
            if (s.TempTableCount > 0) effects.Add("uses temporary staging tables");
            if (s.HasWaitForDelay) effects.Add("includes processing delays for system stability");
            if (s.PrintStatementCount > 5) effects.Add("provides detailed operational logging");
            if (s.WritesTo.Where(t => !IsInfrastructure(t)).Count() >= 3) effects.Add("maintains referential consistency across related tables");
            return effects;
        }

        private static string DetermineProcessingStyle(StructuralSignals s)
        {
            if (s.HasBatchProcessing && s.HasCursorUsage) return "batch with row-by-row processing";
            if (s.HasBatchProcessing) return "batch";
            if (s.HasCursorUsage) return "row-by-row";
            if (s.HasWaitForDelay) return "scheduled";
            if (s.HasTransactionScope) return "transactional";
            return "direct";
        }

        private static string DetermineDataScope(StructuralSignals s)
        {
            if (s.IsReadOnly && s.DistinctTableCount >= 6) return "enterprise-wide";
            if (s.IsReadOnly && s.DistinctTableCount >= 3) return "department-level";
            if (s.DistinctTableCount >= 4) return "cross-domain";
            if (s.DistinctTableCount <= 1) return "single entity";
            return "focused domain";
        }

        public static List<string> SplitCamelCase(string name)
        {
            var words = new List<string>();
            int start = 0;
            for (int i = 1; i < name.Length; i++)
                if (char.IsUpper(name[i]) && !char.IsUpper(name[i - 1]))
                { words.Add(name.Substring(start, i - start)); start = i; }
            if (start < name.Length) words.Add(name.Substring(start));
            return words;
        }

        private static bool IsStopWord(string word) =>
            new HashSet<string>(StringComparer.OrdinalIgnoreCase)
            { "New", "Old", "All", "By", "Top", "Get", "Set", "Bulk", "Monthly", "Daily", "Annual" }
            .Contains(word);

        private static bool IsInfrastructure(string t) =>
            string.IsNullOrEmpty(t) || t.Equals("AuditLog", StringComparison.OrdinalIgnoreCase)
            || t.StartsWith("#") || t.EndsWith("_Archive") || t.Length <= 3;

        public static string HumanizeTableName(string tableName)
        {
            if (string.IsNullOrEmpty(tableName)) return tableName;
            return string.Join(" ", SplitCamelCase(tableName)).ToLower();
        }
    }

    static class SemanticSummaryRenderer
    {
        public static string Render(SemanticIntent intent)
        {
            var sb = new System.Text.StringBuilder();
            sb.Append($"{intent.Operation} {intent.PrimaryEntity}");
            if (intent.RelatedEntities.Any()) sb.Append($" involving {string.Join(", ", intent.RelatedEntities)}");
            sb.Append(". ");
            if (intent.BusinessRules.Any())
            {
                sb.Append("Enforces ");
                sb.Append(string.Join(" with ", intent.BusinessRules));
                sb.Append(". ");
            }
            if (intent.StateTransitions.Any())
            {
                sb.Append(string.Join(". ", intent.StateTransitions.Select(t => char.ToUpper(t[0]) + t.Substring(1))));
                sb.Append(". ");
            }
            if (intent.Outputs.Any())
            {
                sb.Append("Upon completion, ");
                sb.Append(string.Join(" and ", intent.Outputs));
                sb.Append(". ");
            }
            if (intent.SideEffects.Any())
            {
                sb.Append("Additionally, ");
                sb.Append(string.Join(", ", intent.SideEffects));
                sb.Append(". ");
            }
            sb.Append($"Operates as a{("aeiou".Contains(intent.ProcessingStyle[0]) ? "n " : " ")}");
            sb.Append($"{intent.ProcessingStyle} process at {intent.DataScope} scope");
            if (intent.IsAtomic) sb.Append(" with transactional integrity");
            sb.Append(".");
            return sb.ToString();
        }
    }

    class RagChunk
    {
        public string Id { get; set; }
        public string EmbeddingText { get; set; }
        public RagChunkMetadata Metadata { get; set; }
        public RagChunkSql Sql { get; set; }
    }

    class RagChunkMetadata
    {
        public string Classification { get; set; }
        public List<string> Traits { get; set; }
        public double BusinessScore { get; set; }
        public double ReportingScore { get; set; }
        public string Capability { get; set; }
        public string Stage { get; set; }
        public List<string> Reads { get; set; }
        public List<string> Writes { get; set; }
        public List<string> Dependencies { get; set; }
        public int ProcedureLength { get; set; }
        public int LineCount { get; set; }
    }

    class RagChunkSql
    {
        public string Header { get; set; }
        public string Validation { get; set; }
        public string Main { get; set; }
        public string Cleanup { get; set; }
    }

    class RagChunkEmitter
    {
        public List<RagChunk> EmitChunks(List<BusinessDomain> domains, List<StructuralSignals> allSignals)
        {
            var chunks = new List<RagChunk>();
            foreach (var domain in domains)
                foreach (var proc in domain.Procedures)
                    chunks.Add(BuildChunk(proc, domain, allSignals));
            return chunks;
        }

        private RagChunk BuildChunk(StructuralSignals proc, BusinessDomain domain, List<StructuralSignals> allSignals)
        {
            var intent = SemanticIntentExtractor.Extract(proc);
            var stage = GetLifecycleStage(proc);
            return new RagChunk
            {
                Id = proc.ObjectName,
                EmbeddingText = BuildEmbeddingText(proc, domain, intent, stage),
                Metadata = new RagChunkMetadata
                {
                    Classification = proc.ChunkClassification,
                    Traits = proc.Traits,
                    BusinessScore = Math.Round(proc.BusinessLogicScore, 2),
                    ReportingScore = Math.Round(proc.ReportingScore, 2),
                    Capability = domain.DomainName,
                    Stage = stage,
                    Reads = proc.ReadsFrom.Where(t => !IsInfrastructure(t)).OrderBy(t => t).ToList(),
                    Writes = proc.WritesTo.Where(t => !IsInfrastructure(t)).OrderBy(t => t).ToList(),
                    Dependencies = FindDependencies(proc, allSignals),
                    ProcedureLength = proc.RawSql.Length,
                    LineCount = proc.RawSql.Split('\n').Length
                },
                Sql = SplitSqlSections(proc)
            };
        }

        private string BuildEmbeddingText(StructuralSignals proc, BusinessDomain domain, SemanticIntent intent, string stage)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine(proc.SemanticSummary);
            sb.AppendLine();
            sb.AppendLine($"This procedure belongs to the {domain.DomainName} capability");
            sb.AppendLine($"and represents the {stage.ToLower()} stage of the {domain.BusinessFlow.ToLower()} lifecycle.");
            sb.AppendLine();
            sb.AppendLine($"It operates primarily on {intent.PrimaryEntity}");
            if (intent.RelatedEntities.Any()) sb.AppendLine($"and involves {string.Join(", ", intent.RelatedEntities)}.");
            sb.AppendLine();
            if (intent.BusinessRules.Any())
            {
                sb.AppendLine("Business rules enforced:");
                foreach (var rule in intent.BusinessRules) sb.AppendLine($"- {rule}");
                sb.AppendLine();
            }
            if (intent.StateTransitions.Any())
            {
                sb.AppendLine("Lifecycle state management:");
                foreach (var t in intent.StateTransitions) sb.AppendLine($"- {t}");
                sb.AppendLine();
            }
            var related = domain.Procedures.Where(p => p != proc).Select(p => p.ObjectName).Take(5).ToList();
            if (related.Any())
            {
                sb.AppendLine("Related procedures in the same workflow:");
                foreach (var r in related) sb.AppendLine($"- {r}");
                sb.AppendLine();
            }
            sb.Append($"This is a{("aeiou".Contains(intent.ProcessingStyle[0]) ? "n " : " ")}{intent.ProcessingStyle} operation");
            if (intent.IsAtomic) sb.Append(" with transactional guarantees");
            sb.Append(".");
            return sb.ToString().Trim();
        }

        private RagChunkSql SplitSqlSections(StructuralSignals proc)
        {
            string sql = proc.RawSql;
            var sections = new RagChunkSql();
            int headerEnd = FindBoundary(sql, new[] { "BEGIN TRY", "BEGIN TRANSACTION", "BEGIN" });
            sections.Header = headerEnd > 0 ? sql.Substring(0, headerEnd).Trim() : sql.Substring(0, Math.Min(500, sql.Length)).Trim();
            int vs = sql.IndexOf("BEGIN TRY", StringComparison.OrdinalIgnoreCase);
            int ve = FindBoundary(sql, new[] { "INSERT ", "UPDATE ", "DELETE ", "SELECT " }, vs > 0 ? vs : 0);
            if (vs > 0 && ve > vs) sections.Validation = sql.Substring(vs, ve - vs).Trim();
            int ms = ve > 0 ? ve : FindBoundary(sql, new[] { "INSERT ", "UPDATE ", "DELETE ", "SELECT " });
            int me = sql.LastIndexOf("COMMIT TRANSACTION", StringComparison.OrdinalIgnoreCase);
            if (me < 0) me = sql.LastIndexOf("END TRY", StringComparison.OrdinalIgnoreCase);
            sections.Main = (ms > 0 && me > ms) ? sql.Substring(ms, me - ms).Trim() : sql.Trim();
            int cs = sql.LastIndexOf("BEGIN CATCH", StringComparison.OrdinalIgnoreCase);
            if (cs > 0) sections.Cleanup = sql.Substring(cs).Trim();
            return sections;
        }

        private int FindBoundary(string sql, string[] markers, int start = 0)
        {
            int b = int.MaxValue;
            foreach (var m in markers) { int p = sql.IndexOf(m, start, StringComparison.OrdinalIgnoreCase); if (p > 0 && p < b) b = p; }
            return b == int.MaxValue ? -1 : b;
        }

        private List<string> FindDependencies(StructuralSignals proc, List<StructuralSignals> allSignals)
        {
            var deps = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var rt in proc.ReadsFrom.Where(t => !IsInfrastructure(t)))
                foreach (var w in allSignals.Where(o => o != proc && o.WritesTo.Contains(rt) && !o.IsReadOnly).Select(o => o.ObjectName))
                    deps.Add($"{w} (writes {rt})");
            return deps.Take(10).ToList();
        }

        private string GetLifecycleStage(StructuralSignals proc)
        {
            string name = proc.ObjectName.Replace("sp_", "");
            int split = 1;
            while (split < name.Length && !char.IsUpper(name[split])) split++;
            string stage = split > 1 ? name.Substring(0, split) : name;
            if (string.IsNullOrEmpty(stage) || stage.Length <= 1)
                return proc.ChunkClassification switch { "ARCHIVE" => "Archive", "PURGE" => "Purge", "AUDIT" => "Audit", "REPORT" => "Generate", "LOOKUP" => "Get", _ => "Process" };
            return stage;
        }

        private bool IsInfrastructure(string t) =>
            string.IsNullOrEmpty(t) || t.Equals("AuditLog", StringComparison.OrdinalIgnoreCase) || t.StartsWith("#") || t.EndsWith("_Archive") || t.Length <= 3;
    }
    
    // ═══════════════════════════════════════════════════════════════
    // SEMANTIC SECTION CHUNKER
    // ═══════════════════════════════════════════════════════════════

    /// <summary>
    /// A single semantic section of a stored procedure.
    /// </summary>
    class SemanticSection
    {
        public string SectionType { get; set; }       // HEADER, VALIDATION, BUSINESS_RULES, etc.
        public string Purpose { get; set; }    
        public string Summary { get; set; }        // Human-readable description
        public string SqlText { get; set; }            // The SQL code for this section
        public int StartLine { get; set; }
        public int EndLine { get; set; }
    }

    /// <summary>
    /// A fully chunked stored procedure with semantic sections.
    /// </summary>
    class SemanticChunk
    {
        public string Id { get; set; }
        public string EmbeddingText { get; set; }
        public RagChunkMetadata Metadata { get; set; }
        public List<SemanticSection> Sections { get; set; }  // ← ONLY this, no "Sql" property
    }

    class SemanticSectionChunker
    {
        private static readonly HashSet<string> SqlReservedWords = new(StringComparer.OrdinalIgnoreCase)
        {
            "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "IS", "NULL",
            "INSERT", "UPDATE", "DELETE", "INTO", "VALUES", "SET", "AS", "ON",
            "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "CROSS", "FULL",
            "CREATE", "ALTER", "DROP", "TABLE", "VIEW", "PROCEDURE", "FUNCTION",
            "BEGIN", "END", "IF", "ELSE", "THEN", "CASE", "WHEN",
            "DECLARE", "EXEC", "EXECUTE", "RETURN", "PRINT", "RAISERROR", "THROW",
            "COMMIT", "ROLLBACK", "TRANSACTION", "TRAN", "TRY", "CATCH",
            "TOP", "DISTINCT", "GROUP", "HAVING", "ORDER", "BY", "ASC", "DESC",
            "COUNT", "SUM", "AVG", "MIN", "MAX", "EXISTS", "BETWEEN", "LIKE",
            "UNION", "ALL", "OFFSET", "FETCH", "NEXT", "ROWS", "ONLY",
            "SCOPE", "IDENTITY", "NOCOUNT", "READONLY", "OUTPUT"
        };

        private bool IsReservedWord(string token) =>
            SqlReservedWords.Contains(token) || token.StartsWith("@") || token.StartsWith("#");

        private List<string> SplitCamelCase(string name)
        {
            var words = new List<string>();
            int start = 0;
            for (int i = 1; i < name.Length; i++)
                if (char.IsUpper(name[i]) && !char.IsUpper(name[i - 1]))
                { words.Add(name.Substring(start, i - start)); start = i; }
            if (start < name.Length) words.Add(name.Substring(start));
            return words;
        }

        // ═══════════════════════════════════════════════════════════
        // PUBLIC
        // ═══════════════════════════════════════════════════════════

        public SemanticChunk ChunkProcedure(StructuralSignals proc, BusinessDomain domain, List<StructuralSignals> allSignals)
        {
            var intent = SemanticIntentExtractor.Extract(proc);
            var stage = GetLifecycleStage(proc);
            var sections = SplitIntoSections(proc.RawSql, proc);
            sections = MergeRelatedSections(sections);

            return new SemanticChunk
            {
                Id = proc.ObjectName,
                EmbeddingText = BuildEmbeddingText(proc, domain, intent, stage, sections),
                Metadata = new RagChunkMetadata
                {
                    Classification = proc.ChunkClassification,
                    Traits = proc.Traits,
                    BusinessScore = Math.Round(proc.BusinessLogicScore, 2),
                    ReportingScore = Math.Round(proc.ReportingScore, 2),
                    Capability = domain.DomainName,
                    Stage = stage,
                    Reads = proc.ReadsFrom.Where(t => !IsInfrastructure(t)).OrderBy(t => t).ToList(),
                    Writes = proc.WritesTo.Where(t => !IsInfrastructure(t)).OrderBy(t => t).ToList(),
                    Dependencies = FindDependencies(proc, allSignals),
                    ProcedureLength = proc.RawSql.Length,
                    LineCount = proc.RawSql.Split('\n').Length
                },
                Sections = sections
            };
        }

        // ═══════════════════════════════════════════════════════════
        // SECTION SPLITTING
        // ═══════════════════════════════════════════════════════════

        private List<SemanticSection> SplitIntoSections(string sql, StructuralSignals proc)
        {
            var lines = sql.Split('\n');
            var sections = new List<SemanticSection>();
            string currentSection = null;
            int sectionStart = 0;
            var sectionLines = new List<string>();

            for (int i = 0; i < lines.Length; i++)
            {
                string line = lines[i].Trim();
                string lineUpper = line.ToUpperInvariant();

                if (string.IsNullOrWhiteSpace(line) || line.StartsWith("--"))
                { sectionLines.Add(lines[i]); continue; }

                string detected = ClassifyLine(lineUpper, currentSection, i, lines.Length);

                if (detected != currentSection)
                {
                    if (currentSection != null && HasContent(sectionLines))
                        sections.Add(CreateSection(currentSection, sectionLines, sectionStart, i, proc));
                    currentSection = detected;
                    sectionStart = i;
                    sectionLines = new List<string> { lines[i] };
                }
                else sectionLines.Add(lines[i]);
            }

            if (currentSection != null && HasContent(sectionLines))
                sections.Add(CreateSection(currentSection, sectionLines, sectionStart, lines.Length, proc));

            return sections;
        }

        private string ClassifyLine(string lineUpper, string currentSection, int lineIndex, int totalLines)
        {
            if (lineUpper.StartsWith("CREATE PROCEDURE") || lineUpper.StartsWith("CREATE PROC") ||
                lineUpper.StartsWith("ALTER PROCEDURE") || lineUpper.StartsWith("ALTER PROC"))
                return "HEADER";

            if (lineUpper.StartsWith("SET NOCOUNT") && (currentSection == "HEADER" || currentSection == null))
                return "HEADER";

            if (lineUpper.StartsWith("BEGIN CATCH") || lineUpper.StartsWith("END CATCH") ||
                lineUpper.Contains("ERROR_MESSAGE()") || lineUpper.Contains("ERROR_SEVERITY()") ||
                lineUpper.Contains("ERROR_STATE()"))
                return "ERROR_HANDLING";

            if (currentSection == "ERROR_HANDLING") return "ERROR_HANDLING";

            if (lineUpper.StartsWith("DROP TABLE") || lineUpper.StartsWith("DEALLOCATE") || lineUpper.StartsWith("CLOSE "))
                return "CLEANUP";

            if (lineUpper.StartsWith("BEGIN TRANSACTION") || lineUpper.StartsWith("BEGIN TRAN ") ||
                lineUpper == "BEGIN TRAN" || lineUpper.StartsWith("COMMIT") || lineUpper.StartsWith("ROLLBACK"))
                return "TRANSACTION";

            if (lineUpper.Contains("AUDITLOG") && (lineUpper.Contains("INSERT") || lineUpper.Contains("VALUES")))
                return "AUDIT";
            if (currentSection == "AUDIT" && lineUpper.StartsWith("VALUES")) return "AUDIT";

            if (lineUpper.StartsWith("BEGIN TRY")) return "VALIDATION";

            if (lineUpper.StartsWith("DECLARE @"))
                return (currentSection == "HEADER" || currentSection == null) ? "HEADER" : "INITIALIZATION";

            bool beforeFirstDml = currentSection == "VALIDATION" || currentSection == "INITIALIZATION" ||
                                currentSection == "TRANSACTION" || currentSection == "HEADER" || currentSection == null;

            if ((lineUpper.StartsWith("SET @") || lineUpper.StartsWith("SELECT @")) && beforeFirstDml)
                return "INITIALIZATION";

            if ((lineUpper.StartsWith("RAISERROR") || lineUpper.StartsWith("THROW ")) && currentSection != "ERROR_HANDLING")
                return "VALIDATION";

            if ((lineUpper.StartsWith("IF ") || lineUpper.StartsWith("IF(")) && IsInputValidation(lineUpper))
                return "VALIDATION";

            if (lineUpper.StartsWith("INSERT INTO #") || lineUpper.StartsWith("INSERT  INTO #"))
                return "INITIALIZATION";

            // ── BUSINESS_PROCESS: DML + calculations + decisions ─────────
            bool isDml = lineUpper.StartsWith("INSERT ") || lineUpper.StartsWith("UPDATE ") || 
                        lineUpper.StartsWith("DELETE ") || lineUpper.StartsWith("MERGE ");
            bool isCalculation = (lineUpper.StartsWith("SET @") || lineUpper.StartsWith("SELECT @")) && !beforeFirstDml;
            bool isDecision = lineUpper.StartsWith("IF ") || lineUpper.StartsWith("ELSE ") || 
                            lineUpper == "ELSE" || lineUpper.StartsWith("WHILE ");

            if (isDml && !lineUpper.Contains("AUDITLOG") && !lineUpper.Contains("#"))
                return "BUSINESS_PROCESS";
            if (isCalculation || isDecision)
                return "BUSINESS_PROCESS";

            if (lineUpper.StartsWith("SELECT ") && !lineUpper.Contains("SELECT @") &&
                !lineUpper.Contains("SELECT 1") && !lineUpper.Contains("SELECT COUNT"))
            {
                if (currentSection == "DATA_MODIFICATION" || currentSection == "AUDIT" ||
                    currentSection == "TRANSACTION" || lineIndex > totalLines * 0.60)
                    return "OUTPUT";
                return "DATA_RETRIEVAL";
            }

            if (lineUpper.StartsWith("PRINT ") && (currentSection == "DATA_MODIFICATION" ||
                currentSection == "AUDIT" || currentSection == "TRANSACTION" || lineIndex > totalLines * 0.60))
                return "OUTPUT";

            return currentSection ?? "HEADER";
        }

        private bool IsInputValidation(string lineUpper)
        {
            if (lineUpper.Contains("@@TRANCOUNT") || lineUpper.Contains("@@ERROR") || lineUpper.Contains("@@FETCH_STATUS"))
                return false;
            return lineUpper.Contains("EXISTS") || lineUpper.Contains("IS NULL") || lineUpper.Contains("IS NOT NULL") ||
                lineUpper.Contains("!= ") || lineUpper.Contains("<> ") || lineUpper.Contains("NOT IN");
        }

        private bool HasContent(List<string> lines) =>
            lines.Any(l => !string.IsNullOrWhiteSpace(l) && !l.Trim().StartsWith("--"));

        private SemanticSection CreateSection(string type, List<string> lines, int start, int end, StructuralSignals proc)
        {
            var section = new SemanticSection
            {
                SectionType = type,
                Purpose = GetSectionPurpose(type),
                SqlText = string.Join("\n", lines).Trim(),
                StartLine = start + 1,
                EndLine = end
            };
            section.Summary = GenerateSectionSummary(section, proc);
            return section;
        }

        private string GetSectionPurpose(string type) => type switch
        {
            "HEADER" => "Procedure signature and variable declarations",
            "INITIALIZATION" => "Variable initialization and default values",
            "VALIDATION" => "Input validation and business rule checks",
            "BUSINESS_PROCESS" => "Core business logic: calculations, data modifications, workflow progression",
            "DATA_RETRIEVAL" => "Reading existing data from tables",
            "TRANSACTION" => "Transaction control statements",
            "AUDIT" => "Audit logging operations",
            "OUTPUT" => "Returning results to caller",
            "CLEANUP" => "Cleaning up temporary objects",
            "ERROR_HANDLING" => "Exception handling",
            _ => "Other"
        };

        // ═══════════════════════════════════════════════════════════
        // MERGING
        // ═══════════════════════════════════════════════════════════

        private List<SemanticSection> MergeRelatedSections(List<SemanticSection> sections)
        {
            if (sections.Count <= 1) return sections;
            var pass1 = AbsorbSingleLineSections(sections);
            var pass2 = MergeAdjacentSameType(pass1);
            return MergeGloballyByType(pass2);
        }

        private List<SemanticSection> AbsorbSingleLineSections(List<SemanticSection> sections)
        {
            var result = new List<SemanticSection> { sections[0] };
            for (int i = 1; i < sections.Count; i++)
            {
                var cur = sections[i];
                var last = result[result.Count - 1];
                bool lastIsSingle = last.SqlText.Split('\n').Length == 1;
                bool lastIsStructural = last.SectionType == "BUSINESS_PROCESS" || last.SectionType == "AUDIT" ||
                                        last.SectionType == "OUTPUT" || last.SectionType == "ERROR_HANDLING";
                if (lastIsSingle && !lastIsStructural)
                {
                    cur.SqlText = last.SqlText + "\n" + cur.SqlText;
                    cur.StartLine = last.StartLine;
                    result.RemoveAt(result.Count - 1);
                    result.Add(cur);
                }
                else result.Add(cur);
            }
            return result;
        }

        private List<SemanticSection> MergeAdjacentSameType(List<SemanticSection> sections)
        {
            var merged = new List<SemanticSection> { sections[0] };
            var validationTypes = new HashSet<string> { "VALIDATION", "INITIALIZATION" };
            for (int i = 1; i < sections.Count; i++)
            {
                var cur = sections[i];
                var last = merged[merged.Count - 1];
                if (last.SectionType == cur.SectionType)
                { MergeInto(last, cur); continue; }
                if (validationTypes.Contains(last.SectionType) && validationTypes.Contains(cur.SectionType))
                {
                    last.SectionType = "VALIDATION";
                    last.Purpose = "Input validation and business rule checks";
                    MergeInto(last, cur);
                    continue;
                }
                merged.Add(cur);
                // Merge BUSINESS_PROCESS into VALIDATION if it contains validation-like IFs
                // This handles the case where IF/Raiserror pairs get split across sections
                if (last.SectionType == "VALIDATION" && cur.SectionType == "BUSINESS_PROCESS" &&
                    cur.SqlText.Contains("RAISERROR"))
                {
                    MergeInto(last, cur);
                    continue;
                }

                if (last.SectionType == "BUSINESS_PROCESS" && cur.SectionType == "VALIDATION" &&
                    last.SqlText.Contains("IF ") && !last.SqlText.Contains("INSERT") && !last.SqlText.Contains("UPDATE"))
                {
                    last.SectionType = "VALIDATION";
                    last.Purpose = "Input validation and business rule checks";
                    MergeInto(last, cur);
                    continue;
                }
            }
            return merged;
        }

        private List<SemanticSection> MergeGloballyByType(List<SemanticSection> sections)
        {
            var order = new List<string>();
            var groups = new Dictionary<string, SemanticSection>();
            foreach (var s in sections)
            {
                if (!groups.TryGetValue(s.SectionType, out var existing))
                {
                    order.Add(s.SectionType);
                    groups[s.SectionType] = new SemanticSection
                    {
                        SectionType = s.SectionType, Purpose = s.Purpose,
                        Summary = s.Summary, SqlText = s.SqlText,
                        StartLine = s.StartLine, EndLine = s.EndLine
                    };
                }
                else MergeInto(existing, s);
            }
            return order.Select(t => groups[t]).ToList();
        }

        private void MergeInto(SemanticSection target, SemanticSection incoming)
        {
            target.SqlText += "\n" + incoming.SqlText;
            target.StartLine = Math.Min(target.StartLine, incoming.StartLine);
            target.EndLine = Math.Max(target.EndLine, incoming.EndLine);
            target.Summary = JoinSummaries(target.Summary, incoming.Summary);
        }

        private string JoinSummaries(string a, string b)
        {
            if (string.IsNullOrWhiteSpace(a)) return b;
            if (string.IsNullOrWhiteSpace(b)) return a;
            if (a.Equals(b, StringComparison.OrdinalIgnoreCase)) return a;
            var existing = new HashSet<string>(a.Split(';').Select(x => x.Trim()), StringComparer.OrdinalIgnoreCase);
            var added = b.Split(';').Select(x => x.Trim()).Where(x => x.Length > 0 && !existing.Contains(x));
            return added.Any() ? a + "; " + string.Join("; ", added) : a;
        }

        // ═══════════════════════════════════════════════════════════
        // SUMMARIZERS (AST-based)
        // ═══════════════════════════════════════════════════════════

        private string GenerateSectionSummary(SemanticSection section, StructuralSignals proc)
        {
            string sql = section.SqlText;
            return section.SectionType switch
            {
                "HEADER"            => SummarizeHeader(sql, proc),
                "VALIDATION"        => SummarizeValidation(sql),
                "BUSINESS_PROCESS" => SummarizeModification(sql),
                "AUDIT"             => SummarizeAudit(sql),
                "TRANSACTION"       => SummarizeTransaction(sql),
                "OUTPUT"            => SummarizeOutput(sql),
                "DATA_RETRIEVAL"    => SummarizeRetrieval(sql),
                "ERROR_HANDLING"    => SummarizeErrorHandling(sql),
                "CLEANUP"           => SummarizeCleanup(sql),
                "INITIALIZATION"    => SummarizeInitialization(sql),
                _                   => section.Purpose
            };
        }

        private string SummarizeHeader(string sql, StructuralSignals proc)
        {
            var parts = new List<string>();
            string entity = GetBusinessEntity(proc);
            string verb = GetOperationVerb(proc);
            parts.Add($"{verb} {entity}");
            var inputs = ExtractKeyInputs(sql);
            if (inputs.Any()) parts.Add($"requires {string.Join(", ", inputs.Take(4))}");
            if (proc.OutputParameters.Any()) parts.Add($"returns new {entity} identifier");
            return string.Join("; ", parts);
        }

        private string SummarizeValidation(string sql)
        {
            var p = new TSql160Parser(true);
            var f = p.Parse(new StringReader(sql), out IList<ParseError> e);
            if (e.Any() || f == null) return "validates input";
            var x = new ValidationRuleExtractor();
            f.Accept(x);
            return x.BuildSummary();
        }

        private string SummarizeModification(string sql)
        {
            var p = new TSql160Parser(true);
            var f = p.Parse(new StringReader(sql), out IList<ParseError> e);
            if (e.Any() || f == null) return "modifies data";
            var x = new BusinessOperationExtractor();
            f.Accept(x);
            return x.BuildSummary();
        }

        private string SummarizeAudit(string sql)
        {
            var p = new TSql160Parser(true);
            var f = p.Parse(new StringReader(sql), out IList<ParseError> e);
            if (e.Any() || f == null) return "logs operation";
            var x = new AuditDetailExtractor();
            f.Accept(x);
            return x.BuildSummary();
        }

        private string SummarizeTransaction(string sql)
        {
            if (sql.Contains("COMMIT")) return "commits changes";
            if (sql.Contains("ROLLBACK")) return "rolls back on failure";
            if (sql.Contains("BEGIN")) return "starts atomic operation";
            return "controls transaction";
        }

        private string SummarizeOutput(string sql)
        {
            if (sql.Contains("PRINT"))
            {
                if (sql.Contains("success") || sql.Contains("complete")) return "confirms successful completion";
                return "reports status to caller";
            }
            if (sql.Contains("COUNT(*)") || sql.Contains("SUM(")) return "returns summary statistics";
            if (sql.Contains("GROUP BY")) return "returns aggregated report";
            if (sql.Contains("ORDER BY")) return "returns sorted results";
            return "returns query results";
        }

        private string SummarizeRetrieval(string sql)
        {
            var tables = Regex.Matches(sql, @"\bFROM\s+(\w+)", RegexOptions.IgnoreCase)
                .Cast<Match>().Select(m => m.Groups[1].Value)
                .Where(t => !IsReservedWord(t) && !t.StartsWith("#")).Distinct().ToList();
            if (tables.Any() && tables.Count <= 3) return $"reads {string.Join(", ", tables.Select(Humanize))}";
            if (tables.Any()) return $"reads {tables.Count} tables";
            return "retrieves data";
        }

        private string SummarizeErrorHandling(string sql)
        {
            if (sql.Contains("ROLLBACK") && sql.Contains("@@TRANCOUNT")) return "undoes changes and reports error";
            if (sql.Contains("ROLLBACK")) return "reverts changes on error";
            return "handles errors";
        }

        private string SummarizeCleanup(string sql)
        {
            if (sql.Contains("DROP TABLE")) return "removes temporary storage";
            if (sql.Contains("DEALLOCATE")) return "releases cursor resources";
            return "releases resources";
        }

        private string SummarizeInitialization(string sql)
        {
            var a = new List<string>();
            if (sql.Contains("GETDATE()")) a.Add("sets current date");
            if (sql.Contains("MAX(")) a.Add("generates next sequence number");
            if (sql.Contains("COUNT(*)")) a.Add("counts records");
            return a.Any() ? string.Join("; ", a) : "initializes variables";
        }

        // ═══════════════════════════════════════════════════════════
        // BUSINESS ENTITY HELPERS
        // ═══════════════════════════════════════════════════════════

        private string GetBusinessEntity(StructuralSignals proc)
        {
            string name = proc.ObjectName.Replace("sp_", "").Replace("_", "");
            var words = SplitCamelCase(name);
            var entity = string.Join(" ", words.Skip(1).Where(w => w.Length > 2)).ToLower();
            if (!string.IsNullOrEmpty(entity) && entity != "new") return entity;
            var pw = proc.WritesTo.FirstOrDefault(t => IsRealTable(t));
            return pw != null ? Humanize(pw) : "record";
        }

        private string GetOperationVerb(StructuralSignals proc)
        {
            string name = proc.ObjectName.Replace("sp_", "").Replace("_", "");
            var words = SplitCamelCase(name);
            string w = words.FirstOrDefault()?.ToLower() ?? "";
            return w switch
            {
                "add" or "create" or "insert" => "creates", "update" or "modify" or "change" => "updates",
                "delete" or "remove" or "purge" => "removes", "terminate" or "deactivate" => "deactivates",
                "approve" => "approves", "reject" => "rejects", "submit" => "submits",
                "process" or "calculate" => "processes", "generate" or "report" => "generates",
                "get" or "retrieve" or "search" => "retrieves", "record" or "log" => "records",
                "enroll" or "register" => "registers", "transfer" or "move" => "transfers",
                "issue" or "checkout" => "issues", "return" => "returns", "renew" => "renews",
                "reserve" => "reserves", "pay" => "processes payment for", "waive" => "waives",
                "bulk" when words.Count > 1 => $"{words[1].ToLower()}s in bulk",
                _ => "manages"
            };
        }

        private List<string> ExtractKeyInputs(string sql)
        {
            var p = new TSql160Parser(true);
            var f = p.Parse(new StringReader(sql), out IList<ParseError> e);
            if (e.Any() || f == null) return new List<string>();
            var x = new ParameterNameExtractor();
            f.Accept(x);
            return x.Parameters.Where(p => !p.Contains("Error") && !p.Contains("Msg") && !p.StartsWith("New"))
                .Take(4).Select(p => p.Replace("@", "")).Select(HumanizeColumn).ToList();
        }

        // ═══════════════════════════════════════════════════════════
        // NESTED VISITORS
        // ═══════════════════════════════════════════════════════════

        class ValidationRuleExtractor : TSqlFragmentVisitor
        {
            private readonly List<string> _rules = new();
            public override void Visit(BooleanComparisonExpression node)
            {
                string col = GetColumnName(node.FirstExpression);
                string val = GetExpressionText(node.SecondExpression);
                if (!string.IsNullOrEmpty(col))
                {
                    if (node.ComparisonType == BooleanComparisonType.LessThan) _rules.Add($"ensures {HumanizeColumn(col)} below {val}");
                    else if (node.ComparisonType == BooleanComparisonType.GreaterThan) _rules.Add($"ensures {HumanizeColumn(col)} exceeds {val}");
                    else if (node.ComparisonType == BooleanComparisonType.Equals && val.Contains("@")) _rules.Add($"matches {HumanizeColumn(col)} to input");
                }
                base.Visit(node);
            }
            public override void Visit(ExistsPredicate node) { _rules.Add("verifies record exists"); base.Visit(node); }
            public override void Visit(LikePredicate node) { _rules.Add($"validates format of {HumanizeColumn(GetColumnName(node.FirstExpression))}"); base.Visit(node); }
            public override void Visit(InPredicate node) { _rules.Add($"validates {HumanizeColumn(GetColumnName(node.Expression))} is allowed value"); base.Visit(node); }
            public override void Visit(BooleanIsNullExpression node) { _rules.Add($"checks {HumanizeColumn(GetColumnName(node.Expression))} is provided"); base.Visit(node); }
            public string BuildSummary() => _rules.Any() ? string.Join("; ", _rules.Distinct().Take(6)) : "validates input";
        }

        class BusinessOperationExtractor : TSqlFragmentVisitor
        {
            private readonly List<string> _ops = new();
            public override void Visit(InsertStatement node)
            {
                string t = GetTableName(node.InsertSpecification?.Target);
                if (IsBusinessTable(t)) _ops.Add($"creates {Humanize(t)} record");
                base.Visit(node);
            }
            public override void Visit(UpdateStatement node)
            {
                string t = GetTableName(node.UpdateSpecification?.Target);
                if (!IsBusinessTable(t)) { base.Visit(node); return; }
                string entity = Humanize(t);
                if (node.UpdateSpecification?.SetClauses != null)
                    foreach (var c in node.UpdateSpecification.SetClauses)
                        if (c is AssignmentSetClause a) { var d = DescribeBusinessChange(entity, a); if (!string.IsNullOrEmpty(d)) _ops.Add(d); }
                base.Visit(node);
            }
            public override void Visit(DeleteStatement node)
            {
                string t = GetTableName(node.DeleteSpecification?.Target);
                if (IsBusinessTable(t)) _ops.Add($"removes {Humanize(t)} record");
                base.Visit(node);
            }
            private string DescribeBusinessChange(string entity, AssignmentSetClause a)
            {
                string col = GetColumnName(a.Column), ch = HumanizeColumn(col);
                return a.NewValue switch
                {
                    StringLiteral s when IsStatusColumn(col) => DescribeStatusChange(entity, ch, s.Value),
                    IntegerLiteral i when IsActiveColumn(col) => i.Value == "1" ? $"activates {entity}" : $"deactivates {entity}",
                    BinaryExpression b when IsIncrement(b) => $"increases {entity} {ch}",
                    BinaryExpression b when IsDecrement(b) => $"decreases {entity} {ch}",
                    FunctionCall f when IsGetDate(f) && IsModifiedColumn(col) => $"stamps {entity} with current time",
                    FunctionCall f when IsDateAdd(f) => $"extends {entity} {ch}",
                    VariableReference => $"sets {entity} {ch} from input",
                    _ => null
                };
            }
            public string BuildSummary() => _ops.Any() ? string.Join("; ", _ops) : "modifies data";

            static bool IsStatusColumn(string c) => c?.EndsWith("Status", StringComparison.OrdinalIgnoreCase) == true;
            static bool IsActiveColumn(string c) => c?.Equals("IsActive", StringComparison.OrdinalIgnoreCase) == true || c?.Equals("IsDeleted", StringComparison.OrdinalIgnoreCase) == true;
            static bool IsModifiedColumn(string c) => c?.EndsWith("Date", StringComparison.OrdinalIgnoreCase) == true;
            static bool IsIncrement(BinaryExpression b) => b.BinaryExpressionType == BinaryExpressionType.Add;
            static bool IsDecrement(BinaryExpression b) => b.BinaryExpressionType == BinaryExpressionType.Subtract;
            static bool IsGetDate(FunctionCall f) => f.FunctionName?.Value?.Equals("GETDATE", StringComparison.OrdinalIgnoreCase) == true;
            static bool IsDateAdd(FunctionCall f) => f.FunctionName?.Value?.Equals("DATEADD", StringComparison.OrdinalIgnoreCase) == true;

            static string DescribeStatusChange(string entity, string col, string v)
            {
                string l = v.ToLower();
                if (l == "approved") return $"approves {entity}";
                if (l == "rejected") return $"rejects {entity}";
                if (l == "cancelled" || l == "canceled") return $"cancels {entity}";
                if (l == "completed" || l == "done") return $"completes {entity}";
                if (l == "pending") return $"marks {entity} as pending";
                if (l == "active") return $"activates {entity}";
                if (l == "inactive") return $"deactivates {entity}";
                if (l == "returned") return $"marks {entity} as returned";
                if (l == "lost") return $"marks {entity} as lost";
                if (l == "paid") return $"marks {entity} as paid";
                if (l == "unpaid") return $"marks {entity} as unpaid";
                if (l == "waived") return $"waives {entity}";
                if (l == "enrolled") return $"enrolls in {entity}";
                if (l.EndsWith("ed")) return $"marks {entity} as {l}";
                return $"sets {entity} {col} to {l}";
            }
        }

        class AuditDetailExtractor : TSqlFragmentVisitor
        {
            private string _action, _table;
            public override void Visit(InsertStatement node)
            {
                var cols = node.InsertSpecification?.Columns;
                var row = (node.InsertSpecification?.InsertSource as ValuesInsertSource)?.RowValues?.FirstOrDefault()?.ColumnValues;
                if (cols == null || row == null) return;
                for (int i = 0; i < cols.Count && i < row.Count; i++)
                {
                    string cn = cols[i].MultiPartIdentifier?.Identifiers?.LastOrDefault()?.Value ?? "";
                    if (cn.Equals("ActionType", StringComparison.OrdinalIgnoreCase) && row[i] is StringLiteral l) _action = l.Value;
                    if (cn.Equals("TableName", StringComparison.OrdinalIgnoreCase) && row[i] is StringLiteral t) _table = t.Value;
                }
                base.Visit(node);
            }
            public string BuildSummary()
            {
                if (_action != null && _table != null) return $"records {_action.ToLower()} of {Humanize(_table)} to audit trail";
                if (_action != null) return $"logs {_action.ToLower()} operation";
                return "records audit entry";
            }
        }

        class ParameterNameExtractor : TSqlFragmentVisitor
        {
            public List<string> Parameters { get; } = new();
            public override void Visit(VariableReference node) { if (!string.IsNullOrEmpty(node.Name)) Parameters.Add(node.Name); base.Visit(node); }
        }

        // ═══════════════════════════════════════════════════════════
        // SHARED AST HELPERS
        // ═══════════════════════════════════════════════════════════

        static string GetTableName(TableReference t) => (t as NamedTableReference)?.SchemaObject?.BaseIdentifier?.Value ?? "";
        static string GetColumnName(ColumnReferenceExpression c) => c?.MultiPartIdentifier?.Identifiers?.LastOrDefault()?.Value ?? "";
        static string GetColumnName(ScalarExpression e) => (e as ColumnReferenceExpression)?.MultiPartIdentifier?.Identifiers?.LastOrDefault()?.Value ?? "";
        static string GetExpressionText(ScalarExpression e) => e switch { StringLiteral s => $"'{s.Value}'", IntegerLiteral i => i.Value, VariableReference v => v.Name, _ => "value" };
        static bool IsBusinessTable(string t) => !string.IsNullOrEmpty(t) && !t.StartsWith("#") && !t.Equals("AuditLog", StringComparison.OrdinalIgnoreCase) && t.Length > 3;

        static string Humanize(string id)
        {
            if (string.IsNullOrEmpty(id)) return id;
            var w = new List<string>(); int s = 0;
            for (int i = 1; i < id.Length; i++) if (char.IsUpper(id[i]) && !char.IsUpper(id[i - 1])) { w.Add(id.Substring(s, i - s)); s = i; }
            if (s < id.Length) w.Add(id.Substring(s));
            return string.Join(" ", w).ToLower();
        }

        static string HumanizeColumn(string c)
        {
            if (string.IsNullOrEmpty(c)) return c;
            var w = new List<string>(); int s = 0;
            for (int i = 1; i < c.Length; i++) if (char.IsUpper(c[i]) && !char.IsUpper(c[i - 1])) { w.Add(c.Substring(s, i - s)); s = i; }
            if (s < c.Length) w.Add(c.Substring(s));
            return string.Join(" ", w.Select(x => x.Equals("ID", StringComparison.OrdinalIgnoreCase) ? "ID" : x.ToLower()));
        }

        // ═══════════════════════════════════════════════════════════
        // HELPERS
        // ═══════════════════════════════════════════════════════════

        private bool IsRealTable(string t) => !string.IsNullOrEmpty(t) && !t.StartsWith("#") && !t.EndsWith("_cursor") && !t.Equals("AuditLog", StringComparison.OrdinalIgnoreCase) && t.Length > 3;

        private string BuildEmbeddingText(StructuralSignals proc, BusinessDomain domain, SemanticIntent intent, string stage, List<SemanticSection> sections)
        {
            var sb = new System.Text.StringBuilder();
            sb.AppendLine(proc.SemanticSummary);
            sb.AppendLine();
            sb.AppendLine($"This procedure belongs to the {domain.DomainName} capability");
            sb.AppendLine($"and represents the {stage.ToLower()} stage of the {domain.BusinessFlow.ToLower()} lifecycle.");
            sb.AppendLine();
            sb.AppendLine($"It operates primarily on {intent.PrimaryEntity}");
            if (intent.RelatedEntities.Any()) sb.AppendLine($"and involves {string.Join(", ", intent.RelatedEntities)}.");
            sb.AppendLine();
            sb.AppendLine("Semantic sections available:");
            foreach (var st in sections.Select(x => x.SectionType).Distinct()) sb.AppendLine($"- {st}: {GetSectionPurpose(st)}");
            if (intent.BusinessRules.Any()) { sb.AppendLine("Business rules enforced:"); foreach (var r in intent.BusinessRules) sb.AppendLine($"- {r}"); }
            return sb.ToString().Trim();
        }

        private string GetLifecycleStage(StructuralSignals proc)
        {
            string n = proc.ObjectName.Replace("sp_", ""); int sp = 1;
            while (sp < n.Length && !char.IsUpper(n[sp])) sp++;
            string stage = sp > 1 ? n.Substring(0, sp) : n;
            if (string.IsNullOrEmpty(stage) || stage.Length <= 1) return proc.ChunkClassification switch { "ARCHIVE" => "Archive", "PURGE" => "Purge", "AUDIT" => "Audit", "REPORT" => "Generate", _ => "Process" };
            return stage;
        }

        private List<string> FindDependencies(StructuralSignals proc, List<StructuralSignals> allSignals)
        {
            var d = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var rt in proc.ReadsFrom.Where(IsRealTable))
                foreach (var w in allSignals.Where(o => o != proc && o.WritesTo.Contains(rt) && !o.IsReadOnly).Select(o => o.ObjectName))
                    d.Add($"{w} (writes {rt})");
            return d.Take(10).ToList();
        }

        private bool IsInfrastructure(string t) => string.IsNullOrEmpty(t) || t.Equals("AuditLog", StringComparison.OrdinalIgnoreCase) || t.StartsWith("#") || t.EndsWith("_Archive") || t.Length <= 3;
    }



    /// <summary>
    /// Definition of a semantic section in the hierarchy.
    /// </summary>
    class SectionDefinition
    {
        public string Name { get; }
        public string Purpose { get; }
        public string[] TriggerPatterns { get; }     // Patterns that START this section
        public string[] ExcludePatterns { get; }     // Patterns that should NOT trigger this section
        public bool IsValidation { get; }
        public bool IsOutput { get; }
        public string ExcludeAfter { get; }          // Don't start this section after this pattern

        public SectionDefinition(string name, string purpose, string[] triggers, string[] excludes,
            bool isValidation = false, bool isOutput = false, string excludeAfter = null)
        {
            Name = name;
            Purpose = purpose;
            TriggerPatterns = triggers;
            ExcludePatterns = excludes;
            IsValidation = isValidation;
            IsOutput = isOutput;
            ExcludeAfter = excludeAfter;
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
        private readonly TSql160Parser _parser;
        private readonly Sql160ScriptGenerator _generator;

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
                @"\b(?:COUNT|SUM|AVG|MIN|MAX|STRING_AGG|APPROX_COUNT_DISTINCT)\s*\(", RegexOptions.IgnoreCase).Count;
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
        private Vocabulary ExtractVocabulary(List<StructuralSignals> procs)
        {
            var vocab = new Vocabulary();
            foreach (var proc in procs)
            {
                foreach (var table in proc.WritesTo)
                    if (IsRealTable(table)) vocab.AddNoun(table);
                foreach (var table in proc.ReadsFrom)
                    if (IsRealTable(table)) vocab.AddNoun(table);
            }
            foreach (var proc in procs)
            {
                string name = proc.ObjectName.Replace("sp_", "").Replace("_", "");
                var words = SplitCamelCase(name);
                foreach (var word in words)
                {
                    if (word.Length <= 2) continue;
                    if (IsLikelyVerb(word)) vocab.AddVerb(word);
                    else vocab.AddNoun(word);
                }
            }
            foreach (var proc in procs)
                foreach (var (table, columns) in proc.ColumnsRead)
                    foreach (var col in columns)
                        if (IsLikelyStateColumn(col)) vocab.AddStateColumn(col);
            return vocab;
        }

        private bool IsRealTable(string t) =>
            !string.IsNullOrEmpty(t) && !t.StartsWith("#") && !t.EndsWith("_cursor")
            && !t.Equals("AuditLog", StringComparison.OrdinalIgnoreCase) && t.Length > 3;

        private List<string> SplitCamelCase(string name)
        {
            var words = new List<string>();
            int start = 0;
            for (int i = 1; i < name.Length; i++)
                if (char.IsUpper(name[i]) && !char.IsUpper(name[i - 1]))
                { words.Add(name.Substring(start, i - start)); start = i; }
            if (start < name.Length) words.Add(name.Substring(start));
            return words;
        }

        private bool IsLikelyVerb(string word)
        {
            var suffixes = new[] { "ate", "ify", "ize", "ise", "pt", "ed", "ing" };
            return suffixes.Any(s => word.EndsWith(s, StringComparison.OrdinalIgnoreCase)) || word.Length <= 6;
        }

        private bool IsLikelyStateColumn(string col) =>
            col.EndsWith("Status", StringComparison.OrdinalIgnoreCase)
            || col.EndsWith("State", StringComparison.OrdinalIgnoreCase)
            || col.EndsWith("Date", StringComparison.OrdinalIgnoreCase)
            || col.StartsWith("Is", StringComparison.OrdinalIgnoreCase)
            || col.StartsWith("Has", StringComparison.OrdinalIgnoreCase)
            || col.EndsWith("By", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Discovers business capabilities from procedure names, state patterns,
        /// and workflow chains — not from table co-occurrence.
        /// </summary>
        public Dictionary<string, BusinessCapability> DiscoverCapabilities(List<StructuralSignals> procs)
        {
            var capabilities = new Dictionary<string, BusinessCapability>(StringComparer.OrdinalIgnoreCase);

            foreach (var proc in procs)
            {
                var (entities, verbs) = ParseProcedureName(proc.ObjectName);
                string capability = DetermineCapability(proc, entities, verbs);
                string stage = DetermineLifecycleStage(proc, verbs);
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
                cap.CoreEntities.UnionWith(entities);
                if (!cap.LifecycleStages.Contains(stage))
                    cap.LifecycleStages.Add(stage);
                foreach (var neighbor in neighbors)
                    cap.WorkflowEdges.Add((proc.ObjectName, neighbor.ObjectName));
            }

            // Order lifecycle stages within each capability
            foreach (var cap in capabilities.Values)
                cap.LifecycleOrder = OrderLifecycleStages(cap.LifecycleStages);

            // Name capabilities from their dominant entities
            foreach (var cap in capabilities.Values)
                cap.CapabilityName = DeriveCapabilityName(cap);

            // Merge small capabilities
            return MergeSmallCapabilities(capabilities, procs);
        }

        private (List<string> entities, List<string> verbs) ParseProcedureName(string procName)
        {
            string name = procName.Replace("sp_", "").Replace("_", "");
            var words = SplitCamelCase(name);
            
            var entities = new List<string>();
            var verbs = new List<string>();

            // First pass: use extracted vocabulary to classify words
            var vocab = new Vocabulary(); // We'll inject this properly
            foreach (var word in words)
            {
                if (word.Length <= 2) continue;
                if (IsLikelyVerb(word))
                    verbs.Add(word);
                else
                    entities.Add(word);
            }

            // If no verbs found, first word is likely the verb
            if (!verbs.Any() && words.Count > 0)
            {
                verbs.Add(words[0]);
                words.RemoveAt(0);
                foreach (var w in words)
                    if (w.Length > 2 && !verbs.Contains(w))
                        entities.Add(w);
            }

            if (!entities.Any())
                entities.Add("General");

            return (entities, verbs);
        }

        private string DeriveCapabilityName(BusinessCapability cap)
        {
            string entity = cap.CoreEntities
                .OrderByDescending(e => cap.Procedures.Count(p => 
                    p.ObjectName.Contains(e, StringComparison.OrdinalIgnoreCase)))
                .FirstOrDefault() ?? "General";

            var procNames = cap.Procedures.Select(p => p.ObjectName).ToList();
            
            if (procNames.Any(n => n.Contains("Process") || n.Contains("Calculate")))
                return $"{entity}Processing";
            if (procNames.Any(n => n.Contains("Track") || n.Contains("Record") || n.Contains("Attend")))
                return $"{entity}Tracking";
            if (procNames.Any(n => n.Contains("Review") || n.Contains("Perform") || n.Contains("Apprais")))
                return $"{entity}Evaluation";
            if (procNames.Any(n => n.Contains("Train") || n.Contains("Enroll") || n.Contains("Learn")))
                return $"{entity}Development";
            if (procNames.Any(n => n.Contains("Leave") || n.Contains("Request") || n.Contains("Approve")))
                return $"{entity}Management";
            if (procNames.Any(n => n.Contains("Pay") || n.Contains("Salary") || n.Contains("Compens")))
                return $"{entity}Processing";
            
            return $"{entity}Management";
        }

        private Dictionary<string, BusinessCapability> MergeSmallCapabilities(
            Dictionary<string, BusinessCapability> capabilities, List<StructuralSignals> allProcs)
        {
            var merged = new Dictionary<string, BusinessCapability>(StringComparer.OrdinalIgnoreCase);

            foreach (var (name, cap) in capabilities)
            {
                if (cap.Procedures.Count >= 2)
                {
                    merged[name] = cap;
                }
            }

            

            foreach (var (name, cap) in capabilities)
            {
                if (cap.Procedures.Count >= 2) continue;
                var proc = cap.Procedures[0];

                var bestMatch = merged.Values
                    .Where(m => m.Procedures.Any())
                    .OrderByDescending(m =>
                        m.Procedures.SelectMany(p => p.ReadsFrom)
                            .Intersect(proc.ReadsFrom, StringComparer.OrdinalIgnoreCase)
                            .Count(t => IsRealTable(t)))
                    .FirstOrDefault();

                if (bestMatch != null)
                {
                    var overlap = bestMatch.Procedures
                        .SelectMany(p => p.ReadsFrom)
                        .Intersect(proc.ReadsFrom, StringComparer.OrdinalIgnoreCase)
                        .Count(t => IsRealTable(t));

                    if (overlap >= 2 || proc.ChunkClassification == "REPORT")
                    {
                        bestMatch.Procedures.Add(proc);
                        bestMatch.CoreEntities.UnionWith(cap.CoreEntities);
                        continue;
                    }
                }

                if (proc.IsReadOnly || proc.ChunkClassification == "REPORT")
                {
                    if (!merged.ContainsKey("Reporting"))
                        merged["Reporting"] = new BusinessCapability { CapabilityName = "Reporting" };
                    merged["Reporting"].Procedures.Add(proc);
                    merged["Reporting"].CoreEntities.UnionWith(cap.CoreEntities);
                    continue;
                }

                if (proc.ChunkClassification is "ARCHIVE" or "PURGE" or "AUDIT")
                {
                    if (!merged.ContainsKey("Maintenance"))
                        merged["Maintenance"] = new BusinessCapability { CapabilityName = "Maintenance" };
                    merged["Maintenance"].Procedures.Add(proc);
                    merged["Maintenance"].CoreEntities.UnionWith(cap.CoreEntities);
                    continue;
                }

                merged[name] = cap;

            }

            return merged;
        }

        private string DetermineCapability(StructuralSignals proc, List<string> entities, List<string> verbs)
        {
            // Use primary entity from procedure name
            if (entities.Any(e => !e.Equals("General")))
                return entities.First(e => !e.Equals("General"));

            // Use primary write target
            string primaryWrite = proc.WritesTo
                .FirstOrDefault(t => IsRealTable(t));
            
            if (primaryWrite != null)
            {
                foreach (var suffix in new[] { "s", "es", "ies", "Requests", "Reviews", "Records" })
                    if (primaryWrite.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)
                        && primaryWrite.Length - suffix.Length >= 4)
                        return primaryWrite.Substring(0, primaryWrite.Length - suffix.Length);
                return primaryWrite;
            }

            return proc.IsReadOnly ? "Reporting" : "General";
        }

        private string DetermineLifecycleStage(StructuralSignals proc, List<string> verbs)
        {
            if (verbs.Any()) return verbs[0];
            return proc.ChunkClassification switch
            {
                "ARCHIVE" => "Archive",
                "PURGE" => "Purge",
                "AUDIT" => "Audit",
                "REPORT" => "Generate",
                "LOOKUP" => "Get",
                _ => "Process"
            };
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
            if (stages.Count <= 1) return stages.FirstOrDefault() ?? "Process";
            
            // Simple ordering: "create" verbs before "update" before "delete/archive"
            var create = stages.Where(s => 
                s.StartsWith("Add") || s.StartsWith("Create") || s.StartsWith("Submit") 
                || s.StartsWith("Record") || s.StartsWith("Enroll") || s.StartsWith("Issue")
                || s.StartsWith("Reserve")).ToList();
            
            var process = stages.Where(s =>
                s.StartsWith("Approve") || s.StartsWith("Process") || s.StartsWith("Calculate")
                || s.StartsWith("Generate") || s.StartsWith("Renew") || s.StartsWith("Update")
                || s.StartsWith("Transfer") || s.StartsWith("Pay")).ToList();
            
            var close = stages.Where(s =>
                s.StartsWith("Return") || s.StartsWith("Close") || s.StartsWith("Waive")
                || s.StartsWith("Terminate") || s.StartsWith("Delete") || s.StartsWith("Purge")
                || s.StartsWith("Archive") || s.StartsWith("Expire") || s.StartsWith("Cancel")
                || s.StartsWith("Reject")).ToList();
            
            var ordered = new List<string>();
            ordered.AddRange(create);
            ordered.AddRange(process);
            ordered.AddRange(close);
            ordered.AddRange(stages.Where(s => !ordered.Contains(s)));
            
            return string.Join(" → ", ordered.Distinct());
        }
    }

    class BusinessCapability
    {
        public string CapabilityName { get; set; }
        public List<string> LifecycleStages { get; set; } = new();
        public string LifecycleOrder { get; set; }
        public List<StructuralSignals> Procedures { get; set; } = new();
        public HashSet<string> CoreEntities { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public List<(string from, string to)> WorkflowEdges { get; set; } = new();
    }

    class Vocabulary
    {
        private readonly HashSet<string> _verbs = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _nouns = new(StringComparer.OrdinalIgnoreCase);
        private readonly HashSet<string> _stateColumns = new(StringComparer.OrdinalIgnoreCase);

        public void AddVerb(string word) => _verbs.Add(word);
        public void AddNoun(string word) => _nouns.Add(word);
        public void AddStateColumn(string col) => _stateColumns.Add(col);

        public bool IsKnownVerb(string word) => _verbs.Contains(word);
        public bool IsKnownNoun(string word) => _nouns.Contains(word);
        public bool IsStateColumn(string col) => _stateColumns.Contains(col);
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
            // Extract first word after "sp_" as the verb
            string name = proc.ObjectName.Replace("sp_", "");
            
            // Split camelCase to get the first word (verb)
            int firstUpper = 1;
            while (firstUpper < name.Length && !char.IsUpper(name[firstUpper]))
                firstUpper++;
            
            string verb = firstUpper > 1 ? name.Substring(0, firstUpper) : name;
            
            // Use classification as fallback
            if (string.IsNullOrEmpty(verb) || verb.Length <= 1)
            {
                return proc.ChunkClassification switch
                {
                    "ARCHIVE" => "Archive",
                    "PURGE" => "Purge",
                    "AUDIT" => "Audit",
                    "REPORT" => "Generate",
                    "LOOKUP" => "Get",
                    _ => "Execute"
                };
            }
            
            return verb;
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

            // ── Emit RAG chunks ─────────────────────────────────────
            // ── Emit semantic section chunks ─────────────────────────
            var sectionChunker = new SemanticSectionChunker();
            // Should be:
            var chunks = new List<SemanticChunk>();  // ← SemanticChunk, not RagChunk
            foreach (var domain in domains)
            {
                foreach (var proc in domain.Procedures)
                {
                    var chunk = sectionChunker.ChunkProcedure(proc, domain, allSignals);
                    chunks.Add(chunk);
                }
            }


            // After the foreach loop that builds chunks:
            foreach (var chunk in chunks)
            {
                Console.WriteLine($"  {chunk.Id}: {chunk.Sections.Count} sections");
                foreach (var s in chunk.Sections)
                    Console.WriteLine($"    {s.SectionType} (lines {s.StartLine}-{s.EndLine})");
            }

            // ── Save chunks ─────────────────────────────────────────
            string chunksPath = @"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql\shared_data\semantic_chunks.json";
            var chunksJson = JsonSerializer.Serialize(chunks,
                new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase });
            File.WriteAllText(chunksPath, chunksJson);
            Console.WriteLine($"Semantic chunks saved → {chunksPath} ({chunks.Count} chunks)");
        }
    }
}