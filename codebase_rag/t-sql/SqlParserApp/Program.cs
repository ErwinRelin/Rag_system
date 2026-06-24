using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.SqlServer.TransactSql.ScriptDom;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace SqlChunkerApp
{
    // ── Ollama API response models ──────────────────────────────────────
    class OllamaEmbeddingRequest
    {
        public string model { get; set; }
        public string prompt { get; set; }
    }

    class OllamaEmbeddingResponse
    {
        public List<float> embedding { get; set; }
    }

    class SqlChunk
    {
        public int    ChunkId          { get; set; }
        public string FileName         { get; set; }
        public string ObjectType       { get; set; }
        public string ChunkCategory    { get; set; }
        public string ObjectName       { get; set; }
        public string NlDescription    { get; set; }
        public string References       { get; set; }
        public string SqlText          { get; set; }
        public string FullContextBlock { get; set; }
        public float[] Embedding       { get; set; }
    }

    class Program
    {
        // ─── Configuration ───────────────────────────────────────────────
        static string qdrantHost = "localhost";
        static int qdrantPort = 6334;
        static string collectionName = "sql_chunks";
        
        // Ollama configuration
        static string ollamaBaseUrl = "http://localhost:11434";
        static string embeddingModel = "nomic-embed-text";  // Good embedding model
        // Alternative models: "mxbai-embed-large", "all-minilm", "bge-large"
        
        // We'll determine vector size after first embedding call
        static int vectorSize = 768;  // Default for nomic-embed-text
        
        static async Task Main(string[] args)
        {
            string targetFolder = @"C:\Users\Erwin\Desktop\rag_system\codebase_rag\t-sql";

            if (!Directory.Exists(targetFolder))
            {
                Console.WriteLine($"Error: Folder does not exist at '{targetFolder}'");
                return;
            }

            // Initialize HTTP client for Ollama
            using var httpClient = new HttpClient 
            { 
                BaseAddress = new Uri(ollamaBaseUrl),
                Timeout = TimeSpan.FromMiutes(5)
            };
            
            // Verify Ollama is running and model is available
            if (!await VerifyOllamaConnection(httpClient))
            {
                Console.WriteLine("Error: Cannot connect to Ollama. Make sure it's running.");
                return;
            }

            // Get the actual vector size from the model
            vectorSize = await GetEmbeddingSize(httpClient);
            Console.WriteLine($"Using embedding model: {embeddingModel} (vector size: {vectorSize})");

            // Initialize Qdrant client
            var qdrantClient = new QdrantClient(qdrantHost, qdrantPort);
            await SetupQdrantCollection(qdrantClient);

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
                Console.WriteLine($"\n{new string('=', 60)}");
                Console.WriteLine($"PROCESSING: {fileName}");
                Console.WriteLine($"{new string('=', 60)}");

                using var reader = new StreamReader(filePath);
                var fragment = parser.Parse(reader, out IList<ParseError> errors);

                if (errors.Count > 0)
                {
                    Console.WriteLine($"⚠️  Skipped '{fileName}'. Found {errors.Count} syntax error(s):");
                    foreach (var e in errors)
                        Console.WriteLine($"   Line {e.Line}: {e.Message}");
                    continue;
                }

                if (fragment is not TSqlScript script) continue;

                // ── Parse SQL and create chunks ──────────────────────
                var allChunks  = new List<SqlChunk>();
                var ddlChunks  = new List<SqlChunk>();
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
                        string references = ExtractForeignKeyReferences(statement);
                        string nlDesc     = BuildNlDescription(statement, objectType, objectName,
                                                               references, category, rawSql);

                        var chunk = new SqlChunk
                        {
                            ChunkId       = chunkId++,
                            FileName      = fileName,
                            ObjectType    = objectType,
                            ChunkCategory = category,
                            ObjectName    = objectName,
                            NlDescription = nlDesc,
                            References    = references,
                            SqlText       = rawSql
                        };

                        allChunks.Add(chunk);
                        if (category is "DDL" or "VIEW" or "PROCEDURE")
                            ddlChunks.Add(chunk);
                    }
                }

                // Build schema summary
                SqlChunk summaryChunk = BuildSchemaSummaryChunk(fileName, ddlChunks);

                // Build full context blocks
                foreach (var chunk in allChunks)
                    chunk.FullContextBlock = BuildFullContextBlock(chunk);

                // ── Generate embeddings with Ollama ─────────────────
                Console.WriteLine($"Generating embeddings for {allChunks.Count + 1} chunks...");
                
                int processedCount = 0;
                foreach (var chunk in allChunks.Concat(new[] { summaryChunk }))
                {
                    chunk.Embedding = await GenerateOllamaEmbedding(
                        httpClient, 
                        chunk.FullContextBlock
                    );
                    processedCount++;
                    if (processedCount % 5 == 0)
                        Console.WriteLine($"  Generated {processedCount}/{allChunks.Count + 1} embeddings");
                }

                // ── Upload to Qdrant ───────────────────────────────
                Console.WriteLine("Uploading to Qdrant...");
                await UploadChunksToQdrant(qdrantClient, allChunks);
                await UploadChunksToQdrant(qdrantClient, new[] { summaryChunk });

                Console.WriteLine($"✓ Completed: {fileName}");
            }

            Console.WriteLine("\nAll files processed and uploaded to Qdrant!");

            // ✅ CORRECT - at the END, after everything is set up
            await InteractiveQueryLoop(qdrantClient, httpClient);
        }

        // ─────────────────────────────────────────────────────────────────
        // Verify Ollama connection and model availability
        // ─────────────────────────────────────────────────────────────────
        static async Task<bool> VerifyOllamaConnection(HttpClient httpClient)
        {
            try
            {
                var response = await httpClient.GetAsync("/api/tags");
                if (!response.IsSuccessStatusCode) return false;
                
                var content = await response.Content.ReadAsStringAsync();
                var models = JsonSerializer.Deserialize<JsonElement>(content);
                
                // Check if our model exists
                var modelList = models.GetProperty("models").EnumerateArray();
                bool modelExists = modelList.Any(m => 
                    m.GetProperty("name").GetString().StartsWith(embeddingModel));
                
                if (!modelExists)
                {
                    Console.WriteLine($"Model '{embeddingModel}' not found. Pulling it now...");
                    await PullOllamaModel(httpClient);
                }
                
                return true;
            }
            catch
            {
                return false;
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Pull the embedding model if not already available
        // ─────────────────────────────────────────────────────────────────
        static async Task PullOllamaModel(HttpClient httpClient)
        {
            var payload = new { name = embeddingModel };
            var json = JsonSerializer.Serialize(payload);
            var content = new StringContent(json, Encoding.UTF8, "application/json");
            
            var response = await httpClient.PostAsync("/api/pull", content);
            var stream = await response.Content.ReadAsStreamAsync();
            
            using var reader = new StreamReader(stream);
            while (!reader.EndOfStream)
            {
                var line = await reader.ReadLineAsync();
                if (string.IsNullOrEmpty(line)) continue;
                
                var status = JsonSerializer.Deserialize<JsonElement>(line);
                if (status.TryGetProperty("status", out var statusProp))
                    Console.WriteLine($"  Pulling model: {statusProp.GetString()}");
            }
            
            Console.WriteLine("Model pulled successfully!");
        }

        // ─────────────────────────────────────────────────────────────────
        // Get the actual embedding size from the model
        // ─────────────────────────────────────────────────────────────────
        static async Task<int> GetEmbeddingSize(HttpClient httpClient)
        {
            var embedding = await GenerateOllamaEmbedding(httpClient, "test");
            return embedding.Length;
        }

        // ─────────────────────────────────────────────────────────────────
        // Generate embedding using Ollama API
        // ─────────────────────────────────────────────────────────────────
        static async Task<float[]> GenerateOllamaEmbedding(
            HttpClient httpClient, 
            string text)
        {
            try
            {
                var request = new { model = embeddingModel, prompt = text };
                var json = JsonSerializer.Serialize(request);
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await httpClient.PostAsync("/api/embeddings", content);
                response.EnsureSuccessStatusCode();
                
                var responseBody = await response.Content.ReadAsStringAsync();
                var embeddingResponse = JsonSerializer.Deserialize<JsonElement>(responseBody);
                
                var embedding = embeddingResponse
                    .GetProperty("embedding")
                    .EnumerateArray()
                    .Select(e => e.GetSingle())
                    .ToArray();
                
                return embedding;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Error generating embedding: {ex.Message}");
                return new float[vectorSize]; // Return zero vector on error
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Setup Qdrant collection
        // ─────────────────────────────────────────────────────────────────
        static async Task SetupQdrantCollection(QdrantClient client)
        {
            try
            {
                var collections = await client.ListCollectionsAsync();
                if (collections.Any(c => c == collectionName))
                {
                    Console.WriteLine($"Deleting existing collection '{collectionName}'...");
                    await client.DeleteCollectionAsync(collectionName);
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Warning: Could not check/delete collection: {ex.Message}");
            }

            Console.WriteLine($"Creating collection '{collectionName}' with vector size {vectorSize}...");
            await client.CreateCollectionAsync(
                collectionName,
                new VectorParams { 
                    Size = (ulong)vectorSize, 
                    Distance = Distance.Cosine 
                }
            );

            // Create payload indexes for filtering
            Console.WriteLine("Creating payload indexes...");
            await CreatePayloadIndexIfNotExists(client, "chunk_category", PayloadSchemaType.Keyword);
            await CreatePayloadIndexIfNotExists(client, "object_type", PayloadSchemaType.Keyword);
            await CreatePayloadIndexIfNotExists(client, "object_name", PayloadSchemaType.Keyword);
            await CreatePayloadIndexIfNotExists(client, "file_name", PayloadSchemaType.Keyword);
            await CreatePayloadIndexIfNotExists(client, "chunk_id", PayloadSchemaType.Integer);
        }

        static async Task CreatePayloadIndexIfNotExists(
            QdrantClient client, 
            string fieldName, 
            PayloadSchemaType schemaType)
        {
            try
            {
                await client.CreatePayloadIndexAsync(
                    collectionName, 
                    fieldName, 
                    schemaType
                );
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Note: Index on '{fieldName}' might already exist: {ex.Message}");
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Upload chunks to Qdrant
        // ─────────────────────────────────────────────────────────────────
        static async Task UploadChunksToQdrant(
            QdrantClient client, 
            IEnumerable<SqlChunk> chunks)
        {
            var points = new List<PointStruct>();
            
            foreach (var chunk in chunks)
            {
                ulong pointId = GeneratePointId(chunk.FileName, chunk.ChunkId);

                var payload = new Dictionary<string, Value>
                {
                    ["file_name"] = chunk.FileName,
                    ["chunk_id"] = chunk.ChunkId,
                    ["chunk_category"] = chunk.ChunkCategory,
                    ["object_type"] = chunk.ObjectType,
                    ["object_name"] = chunk.ObjectName,
                    ["nl_description"] = chunk.NlDescription,
                    ["references"] = chunk.References ?? "",
                    ["sql_text"] = chunk.SqlText,
                    ["full_context_block"] = chunk.FullContextBlock
                };

                points.Add(new PointStruct
                {
                    Id = pointId,
                    Vectors = chunk.Embedding,
                    Payload = { payload }
                });
            }

            // Upload in batches
            var batchSize = 100;
            for (int i = 0; i < points.Count; i += batchSize)
            {
                var batch = points.Skip(i).Take(batchSize).ToList();
                try
                {
                    await client.UpsertAsync(collectionName, batch);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Error uploading batch: {ex.Message}");
                }
            }

            Console.WriteLine($"  Uploaded {points.Count} chunks successfully");
        }

        // ─────────────────────────────────────────────────────────────────
        // Search function for retrieval
        // ─────────────────────────────────────────────────────────────────
        static async Task<List<SqlChunk>> SearchSimilarChunks(
            QdrantClient client,
            HttpClient httpClient,
            string query,
            int topK = 5,
            string? categoryFilter = null)
        {
            var queryEmbedding = await GenerateOllamaEmbedding(httpClient, query);

            Filter? filter = null;
            if (!string.IsNullOrEmpty(categoryFilter))
            {
                filter = new Filter
                {
                    Must = 
                    {
                        new Condition
                        {
                            Field = new FieldCondition
                            {
                                Key = "chunk_category",
                                Match = new Match { Keyword = categoryFilter }
                            }
                        }
                    }
                };
            }

            var searchResult = await client.SearchAsync(
                collectionName,
                queryEmbedding,  // Use float[] directly, not .ToList()
                filter,
                limit: (ulong)topK
            );

            var results = new List<SqlChunk>();
            foreach (var point in searchResult)
            {
                var payload = point.Payload;
                results.Add(new SqlChunk
                {
                    FileName = payload["file_name"].StringValue,
                    ChunkId = (int)payload["chunk_id"].IntegerValue,
                    ChunkCategory = payload["chunk_category"].StringValue,
                    ObjectType = payload["object_type"].StringValue,
                    ObjectName = payload["object_name"].StringValue,
                    NlDescription = payload["nl_description"].StringValue,
                    References = payload["references"].StringValue,
                    SqlText = payload["sql_text"].StringValue,
                    FullContextBlock = payload["full_context_block"].StringValue
                });
            }

            return results;
        }

        static ulong GeneratePointId(string fileName, int chunkId)
        {
            string combined = $"{fileName}_{chunkId}";
            using var sha256 = System.Security.Cryptography.SHA256.Create();
            var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(combined));
            return BitConverter.ToUInt64(hashBytes, 0);
        }

        // ── Your existing helper methods ────────────────────────────────
        // ── Your existing helper methods ────────────────────────────────
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

        static string ExtractObjectName(TSqlStatement stmt) => stmt switch
        {
            CreateTableStatement     s => s.SchemaObjectName?.BaseIdentifier?.Value ?? "",
            CreateViewStatement      s => s.SchemaObjectName?.BaseIdentifier?.Value ?? "",
            CreateProcedureStatement s => s.ProcedureReference?.Name?.BaseIdentifier?.Value ?? "",
            CreateFunctionStatement  s => s.Name?.BaseIdentifier?.Value ?? "",
            CreateDatabaseStatement  s => s.DatabaseName?.Value ?? "",
            _                          => ""
        };

        static string ExtractForeignKeyReferences(TSqlStatement stmt)
        {
            if (stmt is not CreateTableStatement createTable) return "";

            var refs = new List<string>();
            foreach (var constraint in createTable.Definition.TableConstraints)
            {
                if (constraint is ForeignKeyConstraintDefinition fk)
                {
                    string refTable = fk.ReferenceTableName?.BaseIdentifier?.Value;
                    if (!string.IsNullOrEmpty(refTable))
                        refs.Add(refTable);
                }
            }

            foreach (var col in createTable.Definition.ColumnDefinitions)
            {
                foreach (var constraint in col.Constraints)
                {
                    if (constraint is ForeignKeyConstraintDefinition fk)
                    {
                        string refTable = fk.ReferenceTableName?.BaseIdentifier?.Value;
                        if (!string.IsNullOrEmpty(refTable))
                            refs.Add(refTable);
                    }
                }
            }

            return string.Join(", ", refs.Distinct());
        }

        static string BuildNlDescription(TSqlStatement stmt, string objectType,
                                        string objectName, string references,
                                        string category, string rawSql)
        {
            if (category == "SEED_DATA")
                return $"Sample / seed data INSERT — not schema definition. " +
                    $"Exclude this chunk when only schema context is needed.";

            if (category == "DDL" && objectType == "CreateDatabaseStatement")
                return $"Creates the top-level database named '{objectName}'.";

            if (category == "DDL" && objectType == "UseStatement")
                return "Switches the active database context.";

            if (stmt is CreateTableStatement createTable)
            {
                var cols = createTable.Definition.ColumnDefinitions
                                        .Select(c => c.ColumnIdentifier.Value).ToList();
                string fkNote = string.IsNullOrEmpty(references)
                    ? "No foreign-key dependencies."
                    : $"References: {references}.";
                return $"Defines the '{objectName}' table with columns: {string.Join(", ", cols)}. {fkNote}";
            }

            if (stmt is CreateViewStatement)
                return $"View '{objectName}' — a pre-built SELECT that joins multiple tables " +
                    $"and can be queried directly without rewriting the join logic.";

            if (stmt is CreateProcedureStatement)
                return $"Stored procedure '{objectName}' — encapsulates business logic that " +
                    $"can be called by name; inspect the body for parameters and DML operations.";

            return $"{objectType} statement on object '{objectName}'.";
        }

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
                    if (!string.IsNullOrEmpty(t.References))
                        sb.AppendLine($"    FK → {t.References}");
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

            sb.AppendLine("RELATIONSHIP MAP (table → its FK targets):");
            foreach (var t in tables.Where(t => !string.IsNullOrEmpty(t.References)))
                sb.AppendLine($"  {t.ObjectName} → {t.References}");

            string summaryText = sb.ToString().Trim();

            return new SqlChunk
            {
                ChunkId          = 0,
                FileName         = fileName,
                ObjectType       = "SchemaSummary",
                ChunkCategory    = "SCHEMA_SUMMARY",
                ObjectName       = fileName,
                NlDescription    = "High-level overview of every table, view, and procedure in this file, including FK relationships. Always include this chunk in LLM context.",
                References       = "",
                SqlText          = summaryText,
                FullContextBlock = summaryText
            };
        }

        static string BuildFullContextBlock(SqlChunk chunk)
        {
            var sb = new StringBuilder();
            sb.AppendLine($"[CHUNK]");
            sb.AppendLine($"  File     : {chunk.FileName}");
            sb.AppendLine($"  Id       : {chunk.ChunkId}");
            sb.AppendLine($"  Category : {chunk.ChunkCategory}");
            sb.AppendLine($"  Object   : {chunk.ObjectName}");

            if (!string.IsNullOrEmpty(chunk.References))
                sb.AppendLine($"  FK Refs  : {chunk.References}");

            sb.AppendLine($"  Summary  : {chunk.NlDescription}");
            sb.AppendLine();
            sb.AppendLine("[SQL]");
            sb.AppendLine(chunk.SqlText);
            sb.AppendLine("[/SQL]");

            return sb.ToString().Trim();
        }

        // ─────────────────────────────────────────────────────────────────
        // Ollama chat models
        // ─────────────────────────────────────────────────────────────────
        class OllamaChatRequest
        {
            public string model { get; set; }
            public List<OllamaMessage> messages { get; set; }
            public bool stream { get; set; }
            public OllamaOptions options { get; set; }
        }

        class OllamaMessage
        {
            public string role { get; set; }  // "system", "user", "assistant"
            public string content { get; set; }
        }

        class OllamaOptions
        {
            public float temperature { get; set; }
            public int num_predict { get; set; }
        }

        class OllamaChatResponse
        {
            public string model { get; set; }
            public OllamaMessage message { get; set; }
            public bool done { get; set; }
        }

        // Add these fields to your configuration section
        static string chatModel = "qwen2.5-custom:latest";  // or "mistral", "codellama", etc.
        static int maxTokens = 2048;
        static float temperature = 0.7f;

        // ─────────────────────────────────────────────────────────────────
        // RAG Pipeline: Search → Augment → Generate
        // ─────────────────────────────────────────────────────────────────
        static async Task<string> QueryWithContext(
            QdrantClient qdrantClient,
            HttpClient ollamaClient,
            string userQuery,
            int chunksToRetrieve = 5)
        {
            Console.WriteLine($"\n{new string('=', 60)}");
            Console.WriteLine($"QUERY: {userQuery}");
            Console.WriteLine($"{new string('=', 60)}\n");

            // Step 1: Search for relevant chunks
            Console.WriteLine("Searching for relevant SQL context...");
            var relevantChunks = await SearchSimilarChunks(
                qdrantClient, 
                ollamaClient, 
                userQuery, 
                topK: chunksToRetrieve
            );

            Console.WriteLine($"Found {relevantChunks.Count} relevant chunks\n");

            // Step 2: Build context from retrieved chunks
            var contextBuilder = new StringBuilder();
            contextBuilder.AppendLine("Here are the relevant SQL schema and code chunks:\n");
            
            for (int i = 0; i < relevantChunks.Count; i++)
            {
                var chunk = relevantChunks[i];
                contextBuilder.AppendLine($"--- Chunk {i + 1} ---");
                contextBuilder.AppendLine($"Type: {chunk.ObjectType}");
                contextBuilder.AppendLine($"Category: {chunk.ChunkCategory}");
                contextBuilder.AppendLine($"Name: {chunk.ObjectName}");
                contextBuilder.AppendLine($"Description: {chunk.NlDescription}");
                if (!string.IsNullOrEmpty(chunk.References))
                    contextBuilder.AppendLine($"References: {chunk.References}");
                contextBuilder.AppendLine();
                contextBuilder.AppendLine("SQL Code:");
                contextBuilder.AppendLine(chunk.SqlText);
                contextBuilder.AppendLine();
            }

            string context = contextBuilder.ToString();

            // Step 3: Build the prompt with context
            var systemPrompt = @"You are a SQL expert assistant. You have access to the database schema 
        and SQL code chunks provided in the context. Use this context to answer questions accurately.

        Guidelines:
        - Always reference the specific tables, views, or procedures from the context
        - If the context doesn't contain enough information, say so
        - Explain your reasoning when suggesting SQL queries
        - Include relevant schema details in your answers
        - Format SQL code with proper indentation and syntax highlighting using markdown";

            var userPrompt = $@"Context from the database:
        {context}

        User Question: {userQuery}

        Please provide a detailed answer based on the context above. If you need to write SQL, 
        make sure it aligns with the schema provided.""";
            
            // Step 4: Send to Ollama LLM
            Console.WriteLine("Generating response with Ollama...\n");
            var response = await ChatWithOllama(
                ollamaClient, 
                systemPrompt, 
                userPrompt
            );

            return response;
        }

        // ─────────────────────────────────────────────────────────────────
        // Chat with Ollama (non-streaming)
        // ─────────────────────────────────────────────────────────────────
        static async Task<string> ChatWithOllama(
            HttpClient httpClient,
            string systemPrompt,
            string userPrompt)
        {
            var request = new OllamaChatRequest
            {
                model = chatModel,
                stream = false,
                options = new OllamaOptions
                {
                    temperature = temperature,
                    num_predict = maxTokens
                },
                messages = new List<OllamaMessage>
                {
                    new OllamaMessage { role = "system", content = systemPrompt },
                    new OllamaMessage { role = "user", content = userPrompt }
                }
            };

            try
            {
                var json = JsonSerializer.Serialize(request, new JsonSerializerOptions 
                { 
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase 
                });
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await httpClient.PostAsync("/api/chat", content);
                response.EnsureSuccessStatusCode();
                
                var responseBody = await response.Content.ReadAsStringAsync();
                var chatResponse = JsonSerializer.Deserialize<OllamaChatResponse>(
                    responseBody, 
                    new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.CamelCase }
                );
                
                return chatResponse?.message?.content ?? "No response generated";
            }
            catch (Exception ex)
            {
                return $"Error calling Ollama: {ex.Message}";
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Streaming chat with Ollama (for real-time responses)
        // ─────────────────────────────────────────────────────────────────
        static async Task ChatWithOllamaStreaming(
            HttpClient httpClient,
            string systemPrompt,
            string userPrompt)
        {
            var request = new OllamaChatRequest
            {
                model = chatModel,
                stream = true,
                options = new OllamaOptions
                {
                    temperature = temperature,
                    num_predict = maxTokens
                },
                messages = new List<OllamaMessage>
                {
                    new OllamaMessage { role = "system", content = systemPrompt },
                    new OllamaMessage { role = "user", content = userPrompt }
                }
            };

            try
            {
                var json = JsonSerializer.Serialize(request, new JsonSerializerOptions 
                { 
                    PropertyNamingPolicy = JsonNamingPolicy.CamelCase 
                });
                var content = new StringContent(json, Encoding.UTF8, "application/json");
                
                var response = await httpClient.PostAsync("/api/chat", content);
                response.EnsureSuccessStatusCode();
                
                var stream = await response.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream);
                
                Console.WriteLine("AI Response:\n");
                
                while (!reader.EndOfStream)
                {
                    var line = await reader.ReadLineAsync();
                    if (string.IsNullOrEmpty(line)) continue;
                    
                    try
                    {
                        var chunk = JsonSerializer.Deserialize<JsonElement>(line);
                        if (chunk.TryGetProperty("message", out var message) &&
                            message.TryGetProperty("content", out var messageContent))
                        {
                            Console.Write(messageContent.GetString());
                        }
                    }
                    catch (JsonException)
                    {
                        // Skip malformed JSON chunks
                        continue;
                    }
                }
                
                Console.WriteLine("\n");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error in streaming chat: {ex.Message}");
            }
        }

        // ─────────────────────────────────────────────────────────────────
        // Interactive query loop
        // ─────────────────────────────────────────────────────────────────
        static async Task InteractiveQueryLoop(
            QdrantClient qdrantClient,
            HttpClient ollamaClient)
        {
            Console.WriteLine("\nEntering interactive query mode (type 'exit' to quit):\n");
            
            while (true)
            {
                Console.Write("🔍 Your question: ");
                string query = Console.ReadLine();
                
                if (string.IsNullOrWhiteSpace(query) || query.ToLower() == "exit")
                    break;
                
                try
                {
                    var response = await QueryWithContext(qdrantClient, ollamaClient, query);
                    Console.WriteLine("\n🤖 Answer:\n");
                    Console.WriteLine(response);
                    Console.WriteLine($"\n{new string('─', 60)}\n");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error: {ex.Message}");
                }
            }
        }
    }
}
