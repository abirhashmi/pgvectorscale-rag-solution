DELETE
FROM embeddings
WHERE metadata->>'debug_fault_id' IS NOT NULL;

