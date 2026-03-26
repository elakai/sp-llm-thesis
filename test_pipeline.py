import os
import sys
import time
from unittest.mock import patch
import streamlit as st

mock_session_state = {"session_id": "headless_test_session"}

print("🚀 Booting up Headless Edge-Case Tester (Phase 6: The Final Polish)...")

with patch.object(st, 'session_state', mock_session_state):
    from src.core.retrieval import generate_response

    TEST_SUITES = {
        "1_THE_COURSE_CODE_FIX": [
            # Verifies our Rule 15 successfully triggers a short summary for naked course codes.
            "What is QCPP-512?",
            "QCPP411",
            "What is QCPP452?"
        ]
    }

    output_file = "edge_case_phase6_results.md"

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# 🧪 AXIsstant Phase 6 Edge-Case Results\n\n")

        for suite_name, questions in TEST_SUITES.items():
            print(f"\n[{suite_name}]")
            f.write(f"## {suite_name}\n")
            
            chat_history = [{"role": "assistant", "content": "How can I help you?"}]

            for query in questions:
                print(f"  👉 Asking: {query}")
                f.write(f"**User:** {query}\n\n")
                
                chat_history.append({"role": "user", "content": query})

                full_response = ""
                try:
                    for chunk in generate_response(query, chat_history):
                        full_response += chunk
                        sys.stdout.write(".")
                        sys.stdout.flush()
                except Exception as e:
                    full_response = f"❌ **PIPELINE CRASH:** {str(e)}"

                print(" Done!")
                f.write(f"**AXIsstant:**\n{full_response}\n\n")
                f.write("---\n\n")

                # Truncate tables for memory
                if "|" in full_response and "---" in full_response:
                    table_start = full_response.find("|")
                    full_response = full_response[:table_start] + "\n... [Table truncated]"
                
                chat_history.append({"role": "assistant", "content": full_response})
                time.sleep(1)

    print(f"\n✅ Phase 6 testing complete! Check '{output_file}'.")