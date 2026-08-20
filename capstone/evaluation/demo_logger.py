"""
Demo Example Logger

Captures and formats qualitative examples from evaluation runs.
Logs examples with t_star detection, teacher reasoning, and student outputs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class DemoExampleLogger:
    """
    Logs qualitative examples to a human-readable text file.
    
    Each example includes:
    - Question/prompt
    - Teacher reasoning trace (if available)
    - Detected t_star (disengagement point)
    - Tokens before/after t_star
    - Student output
    - Accuracy result
    
    Highlights "WOW cases" where t_star is early (indicating vision-only sufficiency).
    """

    def __init__(self, output_dir: str | Path = "outputs/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples_file = self.output_dir / "demo_examples.txt"
        self.wow_cases: list[dict] = []
        self._example_count = 0

    def log_example(
        self,
        question: str,
        student_output: str,
        ground_truth: str,
        teacher_trace: Optional[str] = None,
        t_star: Optional[int] = None,
        sequence_length: Optional[int] = None,
        before_t_star: Optional[str] = None,
        after_t_star: Optional[str] = None,
        is_correct: bool = False,
        benchmark_name: str = "unknown",
    ) -> None:
        """
        Log a single evaluation example.
        
        Args:
            question: The input question/prompt
            student_output: The model's prediction
            ground_truth: Expected answer
            teacher_trace: Optional teacher reasoning text
            t_star: Token index where disengagement detected
            sequence_length: Total sequence length
            before_t_star: Token representation before t_star
            after_t_star: Token representation after t_star
            is_correct: Whether the student got it right
            benchmark_name: Which benchmark this is from
        """
        self._example_count += 1
        
        # Detect "WOW" cases: t_star very early relative to sequence
        is_wow_case = False
        if t_star is not None and sequence_length is not None:
            early_ratio = t_star / max(sequence_length, 1)
            if early_ratio < 0.2 and sequence_length > 20:  # Early cutoff on substantial output
                is_wow_case = True
                self.wow_cases.append({
                    "example_num": self._example_count,
                    "t_star": t_star,
                    "seq_len": sequence_length,
                    "question": question[:100],
                })

        # Format the example
        example_text = self._format_example(
            example_num=self._example_count,
            question=question,
            student_output=student_output,
            ground_truth=ground_truth,
            teacher_trace=teacher_trace,
            t_star=t_star,
            sequence_length=sequence_length,
            before_t_star=before_t_star,
            after_t_star=after_t_star,
            is_correct=is_correct,
            is_wow_case=is_wow_case,
            benchmark_name=benchmark_name,
        )

        # Append to file
        with open(self.examples_file, "a", encoding="utf-8") as f:
            f.write(example_text)
            f.write("\n" + "=" * 80 + "\n\n")

    def finalize(self) -> None:
        """Write summary and WOW cases to the file."""
        with open(self.examples_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Total examples logged: {self._example_count}\n")
            f.write(f"WOW cases (early t_star): {len(self.wow_cases)}\n\n")
            
            if self.wow_cases:
                f.write("WOW CASES (Early Vision Cutoff):\n")
                f.write("-" * 80 + "\n")
                for wow in self.wow_cases[:5]:  # Top 5 wow cases
                    f.write(
                        f"Example #{wow['example_num']}: "
                        f"t_star={wow['t_star']} / {wow['seq_len']} tokens "
                        f"({100*wow['t_star']/max(wow['seq_len'], 1):.1f}%)\n"
                    )
                    f.write(f"  Question: {wow['question']}...\n\n")

        logger.info(
            "Demo examples saved to %s (%d examples, %d WOW cases)",
            self.examples_file,
            self._example_count,
            len(self.wow_cases),
        )

    @staticmethod
    def _format_example(
        example_num: int,
        question: str,
        student_output: str,
        ground_truth: str,
        teacher_trace: Optional[str],
        t_star: Optional[int],
        sequence_length: Optional[int],
        before_t_star: Optional[str],
        after_t_star: Optional[str],
        is_correct: bool,
        is_wow_case: bool,
        benchmark_name: str,
    ) -> str:
        """Format a single example as human-readable text."""
        lines = []
        
        if is_wow_case:
            lines.append("*** WOW CASE: Early vision cutoff ***\n")
        
        lines.append(f"Example #{example_num}")
        lines.append(f"Benchmark: {benchmark_name}")
        lines.append(f"Result: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}\n")
        
        lines.append("Question:")
        lines.append(question + "\n")
        
        if teacher_trace:
            lines.append("Teacher Reasoning:")
            lines.append(teacher_trace + "\n")
        
        if t_star is not None:
            lines.append(f"Detected t_star (disengagement point): {t_star}")
            if sequence_length is not None:
                pct = 100 * t_star / max(sequence_length, 1)
                lines.append(f"  Position in sequence: {t_star}/{sequence_length} tokens ({pct:.1f}%)\n")
        
        if before_t_star:
            lines.append("Before t_star:")
            lines.append(before_t_star + "\n")
        
        if after_t_star:
            lines.append("After t_star:")
            lines.append(after_t_star + "\n")
        
        lines.append("Ground Truth:")
        lines.append(ground_truth + "\n")
        
        lines.append("Student Output:")
        lines.append(student_output + "\n")
        
        return "\n".join(lines)
