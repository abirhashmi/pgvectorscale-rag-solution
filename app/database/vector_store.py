# app/database/vector_store.py
import logging, os, sys, time
from typing import Any, List, Optional, Tuple, Union
from datetime import datetime

import psycopg2
import pandas as pd
from dotenv import load_dotenv

# Load .env and override any stale OS/user vars
load_dotenv(override=True)

from openai import OpenAI
from timescale_vector import client
from config.settings import get_settings

class VectorStore:
    """Manage embeddings, storage, and similarity search against Timescale + pgvector."""

    def __init__(self):
        """Initialize settings, OpenAI client, and Timescale Vector client."""
        self.settings = get_settings()

        # ---- OpenAI ----
        self.openai_client = OpenAI(api_key=self.settings.openai.api_key)
        self.embedding_model = self.settings.openai.embedding_model

        # ---- DB / table config (cached for fallbacks) ----
        self.vector_settings = self.settings.vector_store  # keep for external callers
        self.table_name: str = self.vector_settings.table_name
        self.num_dimensions: int = self.vector_settings.embedding_dimensions
        self.time_partition_interval = self.vector_settings.time_partition_interval

        # Build DSN: **prefer ENV**, then settings
        env_dsn = (
            os.getenv("VECTOR_SERVICE_URL")
            or os.getenv("DATABASE_URL")
            or os.getenv("TIME_SCALE_SERVICE_URL")
            or os.getenv("TIMESCALE_SERVICE_URL")
        )
        settings_dsn = getattr(self.settings.database, "service_url", None)

        print(f"▶ raw .env URI:  {env_dsn}", file=sys.stderr)
        print(f"▶ settings URI:  {settings_dsn}", file=sys.stderr)

        self.service_url: str = env_dsn or settings_dsn
        if not self.service_url:
            raise RuntimeError(
                "No DB DSN found. Set VECTOR_SERVICE_URL / DATABASE_URL / TIME_SCALE_SERVICE_URL "
                "or provide database.service_url in your settings."
            )

        # cosine is typical for OpenAI embeddings
        self.pgvector_ops = "vector_cosine_ops"

        # Helpful to see which DSN is actually used at runtime
        print(f"[VectorStore] Using DSN: {self.service_url}", file=sys.stderr)

        # ---- Timescale Vector sync client (works fine without vectorscale) ----
        self.vec_client = client.Sync(
            self.service_url,
            self.table_name,
            self.num_dimensions,
            time_partition_interval=self.time_partition_interval,
        )

    # --------------------------
    # Embeddings
    # --------------------------
    def get_embedding(self, text: str) -> List[float]:
        """Return an embedding for the given text."""
        text = text.replace("\n", " ")
        start = time.time()
        resp = self.openai_client.embeddings.create(input=[text], model=self.embedding_model)
        emb = resp.data[0].embedding
        logging.info("Embedding generated in %.3f seconds", time.time() - start)
        return emb

    # --------------------------
    # Table / Index management
    # --------------------------
    def create_tables(self) -> None:
        """
        Try the library helper (which may attempt vectorscale). If that fails due to
        'vectorscale' not being installed, fall back to a plain pgvector schema + HNSW index.
        """
        try:
            self.vec_client.create_tables()
            return
        except Exception as e:
            if "vectorscale" not in str(e).lower():
                # Not a vectorscale problem; surface it
                raise

        logging.warning("vectorscale not available; creating plain pgvector schema instead.")
        with psycopg2.connect(self.service_url) as conn, conn.cursor() as cur:
            # Ensure extensions
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

            # Create table matching expected schema
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY,
                    metadata JSONB,
                    contents TEXT,
                    embedding VECTOR({self.num_dimensions}),
                    created_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )

            # Create ANN index using pgvector HNSW
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw
                ON {self.table_name}
                USING hnsw (embedding {self.pgvector_ops});
                """
            )
            conn.commit()

    def create_index(self) -> None:
        """(pgvector) Ensure an HNSW index exists on the embedding column."""
        with psycopg2.connect(self.service_url) as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_embedding_hnsw
                ON {self.table_name}
                USING hnsw (embedding {self.pgvector_ops});
                """
            )
            conn.commit()

    def drop_index(self) -> None:
        """Drop the HNSW index (pgvector)."""
        with psycopg2.connect(self.service_url) as conn, conn.cursor() as cur:
            cur.execute(f"DROP INDEX IF EXISTS {self.table_name}_embedding_hnsw;")
            conn.commit()

    # --------------------------
    # Data ops
    # --------------------------
    def upsert(self, df: pd.DataFrame) -> None:
        """
        Insert or update records from a pandas DataFrame.
        Expected columns: id, metadata, contents, embedding
        """
        records = df.to_records(index=False)
        self.vec_client.upsert(list(records))
        logging.info("Inserted %d records into %s", len(df), self.table_name)

    # --------------------------
    # Search
    # --------------------------
    def search(
        self,
        query_text: str,
        limit: int = 5,
        metadata_filter: Union[dict, List[dict]] = None,
        predicates: Optional[client.Predicates] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        return_dataframe: bool = True,
    ) -> Union[List[Tuple[Any, ...]], pd.DataFrame]:
        """
        Query the vector database for similar embeddings based on input text.
        """
        query_embedding = self.get_embedding(query_text)
        start = time.time()

        search_args = {"limit": limit}
        if metadata_filter:
            search_args["filter"] = metadata_filter
        if predicates:
            search_args["predicates"] = predicates
        if time_range:
            start_date, end_date = time_range
            search_args["uuid_time_filter"] = client.UUIDTimeRange(start_date, end_date)

        results = self.vec_client.search(query_embedding, **search_args)
        logging.info("Vector search completed in %.3f seconds", time.time() - start)

        return self._create_dataframe_from_results(results) if return_dataframe else results

    # alias used elsewhere in your code
    query = search

    # --------------------------
    # Helpers
    # --------------------------
    def _create_dataframe_from_results(
        self, results: List[Tuple[Any, ...]]
    ) -> pd.DataFrame:
        """
        Convert tuple results -> DataFrame with expanded metadata.
        Library typically returns tuples like: (id, metadata, content, embedding, distance)
        """
        if not results:
            return pd.DataFrame(columns=["id", "content", "distance"])

        # Name columns explicitly; adjust if your client returns a different tuple shape
        cols = ["id", "metadata", "content", "embedding", "distance"]
        df = pd.DataFrame(results, columns=cols)

        # Expand metadata JSON to top-level columns if present
        if "metadata" in df.columns:
            meta = df["metadata"].apply(lambda m: m or {})
            df = pd.concat([df.drop(columns=["metadata"]), meta.apply(pd.Series)], axis=1)

        # id to string for readability
        if "id" in df.columns:
            df["id"] = df["id"].astype(str)

        return df

    # --------------------------
    # Delete ops
    # --------------------------
    def delete(
        self,
        ids: List[str] = None,
        metadata_filter: dict = None,
        delete_all: bool = False,
    ) -> None:
        """Delete records by IDs, metadata filter, or delete_all (choose exactly one)."""
        if sum(bool(x) for x in (ids, metadata_filter, delete_all)) != 1:
            raise ValueError("Provide exactly one of: ids, metadata_filter, or delete_all")

        if delete_all:
            self.vec_client.delete_all()
            logging.info("Deleted ALL records from %s", self.table_name)
        elif ids:
            self.vec_client.delete_by_ids(ids)
            logging.info("Deleted %d records from %s", len(ids), self.table_name)
        elif metadata_filter:
            self.vec_client.delete_by_metadata(metadata_filter)
            logging.info("Deleted records matching metadata filter from %s", self.table_name)
