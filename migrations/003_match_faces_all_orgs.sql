-- Function: match_faces across ALL organizations
-- Used by face-login when org_id is not specified
CREATE OR REPLACE FUNCTION match_faces_all_orgs(
    query_embedding vector(512),
    match_threshold float,
    match_count int
)
RETURNS TABLE (
    id uuid,
    name text,
    similarity float,
    org_id uuid
)
LANGUAGE plpgsql STABLE
AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.id,
        f.name,
        1 - (f.embedding <=> query_embedding) AS similarity,
        f.org_id
    FROM faces f
    WHERE 1 - (f.embedding <=> query_embedding) > match_threshold
    ORDER BY f.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
