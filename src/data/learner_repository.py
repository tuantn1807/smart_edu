"""
SQLite-backed Learner State Repository for Agent Memory Persistence.
Stores and loads student LearnerState, mastery levels, misconception history, learning path, and chat logs across sessions.
"""

import sqlite3
import json
import os
from typing import Dict, List, Any, Optional
from src.core.learner_state import LearnerState, MisconceptionRecord, LearningPathStep


class LearnerStateRepository:
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = "data/learner_state.db"
        self.set_db_path(db_path)

    def set_db_path(self, db_path: str):
        self.db_path = db_path
        if self.db_path != ":memory:":
            abs_path = os.path.abspath(self.db_path)
            dir_name = os.path.dirname(abs_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            self._conn = None
        else:
            self._conn = sqlite3.connect(":memory:")
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _close_connection(self, conn: sqlite3.Connection):
        if self._conn is None:
            conn.close()

    def _init_db(self):
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learner_states (
                        student_id TEXT PRIMARY KEY,
                        student_name TEXT,
                        consecutive_correct INTEGER DEFAULT 0,
                        consecutive_incorrect INTEGER DEFAULT 0,
                        answered_questions TEXT,
                        created_at TEXT,
                        updated_at TEXT
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS concept_mastery (
                        student_id TEXT,
                        concept_id TEXT,
                        score REAL,
                        updated_at TEXT,
                        PRIMARY KEY (student_id, concept_id),
                        FOREIGN KEY (student_id) REFERENCES learner_states(student_id) ON DELETE CASCADE
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS misconceptions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT,
                        concept_id TEXT,
                        misconception_name TEXT,
                        description TEXT,
                        detected_at TEXT,
                        severity TEXT,
                        resolved INTEGER DEFAULT 0,
                        FOREIGN KEY (student_id) REFERENCES learner_states(student_id) ON DELETE CASCADE
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_path_steps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT,
                        step_id INTEGER,
                        concept_id TEXT,
                        concept_name TEXT,
                        action_type TEXT,
                        description TEXT,
                        status TEXT,
                        question_id TEXT,
                        question_details TEXT,
                        FOREIGN KEY (student_id) REFERENCES learner_states(student_id) ON DELETE CASCADE
                    );
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS interaction_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        student_id TEXT,
                        timestamp TEXT,
                        role TEXT,
                        agent TEXT,
                        message TEXT,
                        metadata TEXT,
                        FOREIGN KEY (student_id) REFERENCES learner_states(student_id) ON DELETE CASCADE
                    );
                """)
        finally:
            self._close_connection(conn)

    def save_learner_state(self, state: LearnerState):
        """Persist a LearnerState instance into the SQLite database."""
        conn = self._get_connection()
        try:
            with conn:
                # 1. Upsert main learner_state table
                conn.execute("""
                    INSERT OR REPLACE INTO learner_states (
                        student_id, student_name, consecutive_correct, consecutive_incorrect,
                        answered_questions, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.student_id,
                    state.student_name,
                    state.consecutive_correct,
                    state.consecutive_incorrect,
                    json.dumps(state.answered_questions),
                    state.created_at,
                    state.updated_at
                ))

                # 2. Update concept_mastery
                conn.execute("DELETE FROM concept_mastery WHERE student_id = ?", (state.student_id,))
                for concept_id, score in state.mastery_levels.items():
                    conn.execute("""
                        INSERT INTO concept_mastery (student_id, concept_id, score, updated_at)
                        VALUES (?, ?, ?, ?)
                    """, (state.student_id, concept_id, float(score), state.updated_at))

                # 3. Update misconceptions
                conn.execute("DELETE FROM misconceptions WHERE student_id = ?", (state.student_id,))
                for m in state.misconceptions:
                    conn.execute("""
                        INSERT INTO misconceptions (
                            student_id, concept_id, misconception_name, description, detected_at, severity, resolved
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        state.student_id,
                        m.concept_id,
                        m.misconception_name,
                        m.description,
                        m.detected_at,
                        m.severity,
                        1 if m.resolved else 0
                    ))

                # 4. Update learning_path_steps
                conn.execute("DELETE FROM learning_path_steps WHERE student_id = ?", (state.student_id,))
                for step in state.active_learning_path:
                    conn.execute("""
                        INSERT INTO learning_path_steps (
                            student_id, step_id, concept_id, concept_name, action_type, description, status, question_id, question_details
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        state.student_id,
                        step.step_id,
                        step.concept_id,
                        step.concept_name,
                        step.action_type,
                        step.description,
                        step.status,
                        step.question_id,
                        json.dumps(step.question_details) if step.question_details is not None else None
                    ))

                # 5. Update interaction_history
                conn.execute("DELETE FROM interaction_history WHERE student_id = ?", (state.student_id,))
                for entry in state.interaction_history:
                    conn.execute("""
                        INSERT INTO interaction_history (
                            student_id, timestamp, role, agent, message, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        state.student_id,
                        entry.get("timestamp", ""),
                        entry.get("role", ""),
                        entry.get("agent", ""),
                        entry.get("message", ""),
                        json.dumps(entry.get("metadata", {}))
                    ))
        finally:
            self._close_connection(conn)

    def load_learner_state(self, student_id: str) -> Optional[LearnerState]:
        """Load and reconstruct a LearnerState instance from the SQLite database."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT student_name, consecutive_correct, consecutive_incorrect, answered_questions, created_at, updated_at
                FROM learner_states WHERE student_id = ?
            """, (student_id,))
            row = cursor.fetchone()
            if not row:
                return None

            student_name, cons_corr, cons_inc, ans_q_json, created_at, updated_at = row
            state = LearnerState(student_id=student_id, student_name=student_name)
            state.consecutive_correct = cons_corr
            state.consecutive_incorrect = cons_inc
            state.answered_questions = json.loads(ans_q_json) if ans_q_json else []
            state.created_at = created_at
            state.updated_at = updated_at

            # Load concept_mastery
            cursor.execute("SELECT concept_id, score FROM concept_mastery WHERE student_id = ?", (student_id,))
            for cid, score in cursor.fetchall():
                state.mastery_levels[cid] = float(score)

            # Load misconceptions
            cursor.execute("""
                SELECT concept_id, misconception_name, description, detected_at, severity, resolved
                FROM misconceptions WHERE student_id = ? ORDER BY id ASC
            """, (student_id,))
            state.misconceptions = []
            for cid, mname, desc, det_at, sev, res in cursor.fetchall():
                state.misconceptions.append(MisconceptionRecord(
                    concept_id=cid,
                    misconception_name=mname,
                    description=desc,
                    detected_at=det_at,
                    severity=sev,
                    resolved=bool(res)
                ))

            # Load learning_path_steps
            cursor.execute("""
                SELECT step_id, concept_id, concept_name, action_type, description, status, question_id, question_details
                FROM learning_path_steps WHERE student_id = ? ORDER BY step_id ASC
            """, (student_id,))
            state.active_learning_path = []
            for sid, cid, cname, act, desc, stat, qid, qdet_json in cursor.fetchall():
                qdet = json.loads(qdet_json) if qdet_json else None
                state.active_learning_path.append(LearningPathStep(
                    step_id=sid,
                    concept_id=cid,
                    concept_name=cname,
                    action_type=act,
                    description=desc,
                    status=stat,
                    question_id=qid,
                    question_details=qdet
                ))

            # Load interaction_history
            cursor.execute("""
                SELECT timestamp, role, agent, message, metadata
                FROM interaction_history WHERE student_id = ? ORDER BY id ASC
            """, (student_id,))
            state.interaction_history = []
            for ts, role, agent, msg, meta_json in cursor.fetchall():
                meta = json.loads(meta_json) if meta_json else {}
                state.interaction_history.append({
                    "timestamp": ts,
                    "role": role,
                    "agent": agent,
                    "message": msg,
                    "metadata": meta
                })

            return state
        finally:
            self._close_connection(conn)

    def delete_learner_state(self, student_id: str):
        """Remove all persistence records for a student."""
        conn = self._get_connection()
        try:
            with conn:
                conn.execute("DELETE FROM learner_states WHERE student_id = ?", (student_id,))
                conn.execute("DELETE FROM concept_mastery WHERE student_id = ?", (student_id,))
                conn.execute("DELETE FROM misconceptions WHERE student_id = ?", (student_id,))
                conn.execute("DELETE FROM learning_path_steps WHERE student_id = ?", (student_id,))
                conn.execute("DELETE FROM interaction_history WHERE student_id = ?", (student_id,))
        finally:
            self._close_connection(conn)

    def list_student_ids(self) -> List[str]:
        """Get list of stored student IDs."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT student_id FROM learner_states ORDER BY updated_at DESC")
            rows = cursor.fetchall()
            return [r[0] for r in rows]
        finally:
            self._close_connection(conn)
