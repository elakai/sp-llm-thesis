# suggested_questions.py — keep this file but simplify it

import re
from typing import List
import streamlit as st


def render_suggested_questions(questions: List[str], key_prefix: str) -> None:
    if not questions:
        return
    st.markdown("<div class='suggested-title'>You might also ask</div>", unsafe_allow_html=True)
    for idx, question in enumerate(questions):
        if st.button(question, key=f"{key_prefix}_{idx}"):
            st.session_state["queued_query"] = question
            st.rerun()