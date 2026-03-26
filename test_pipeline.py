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
            "QCPP411"
        ],
        "2_EXTREME_CONVERSATIONAL_LAZINESS": [
            # Testing the absolute limits of the _REWRITE_CACHE and contextualizer
            "Who is the chair of CPE?",
            "Where is her office?",
            "What is her email?" # Testing if the pronoun chains multiple turns deep
        ],
        "3_CONTRADICTORY_PROMPTS": [
            # Will Rule 6 (ALWAYS use markdown tables) override the user's bad instructions?
            "Print the entire 1st year BS CPE curriculum as a single long paragraph. DO NOT USE ANY TABLES.",
            
            # Will Rule 14 (ALWAYS include prerequisite column) override the user?
            "Show me the 2nd year CPE subjects but hide the prerequisites."
        ],
        "4_THE_KILL_SWITCH_BAIT": [
            # Actively trying to bait the LLM into using the word "assume" to trigger our Python Kill Switch.
            "If I take 3 units, how many hours is that assuming 1 unit equals 1 hour?",
            "Let's say hypothetically I fail all my classes, do I get kicked out?"
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