"""Bounded, same-image conversation memory for Visual-JEV inference.

The memory keeps complete turn records but retrieves only a small, relevant
subset for each new question.  This avoids blindly concatenating every turn,
which is both expensive and prone to history shortcut bias.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


HistoryMode = Literal["off", "recent", "auto", "legacy-auto"]
HistoryPromptPolicy = Literal["adaptive", "legacy-image-only"]

_ENGLISH_STOPWORDS = {
    "a", "an", "and", "are", "do", "does", "for", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "what", "where", "which", "who",
}
_CJK_STOP_CHARS = set("这是的了吗呢啊个一有和在为什麼什么")
_COREFERENCE_CUES = (
    " it ", " its ", " they ", " them ", " their ", " he ", " him ",
    " his ", " she ", " her ", " this ", " that ", " these ", " those ",
    "former", "latter", "previous", "above", "same one",
    "它", "牠", "他们", "她们", "它们", "他", "她", "这个", "那个",
    "这些", "那些", "前面", "刚才", "之前", "上述", "同一个", "该",
)

# These cues ask about the dialogue itself, rather than an entity that merely
# happens to share words with an earlier turn.  Lexical-overlap retrieval
# cannot handle questions such as ``上一个问题是什么？`` because the answer is
# precisely the earlier wording, which is absent from the query.
_LOCAL_HISTORY_REFERENCE_PATTERNS = (
    r"\b(?:last|previous|prior|preceding)\s+(?:question|turn|message|answer)\b",
    r"\bwhat\s+did\s+(?:i|we|you)\s+(?:just\s+|previously\s+|last\s+)?(?:ask|say|mention|discuss)\b",
    r"(?:上一个|前一个|上一条|前一条|上一轮|前一轮|上一问|前一问).{0,6}(?:问题|提问|消息|回答|问|说)",
    r"(?:刚才|方才|之前).{0,8}(?:问了?|说了?|提到|回答).{0,4}(?:什么|啥|哪)",
    r"(?:我|你|我们).{0,4}(?:刚才|之前).{0,5}(?:问了?|说了?|提到)",
)
_GLOBAL_HISTORY_REFERENCE_PATTERNS = (
    r"\b(?:conversation|chat|dialogue)\s+history\b",
    r"\b(?:earlier|previous)\s+(?:conversation|dialogue|discussion)\b",
    r"\b(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+turn\b",
    r"(?:历史对话|对话历史|聊天记录|会话记录|上下文记录)",
    r"(?:第[一二三四五六七八九十\d]+轮|第[一二三四五六七八九十\d]+个问题)",
)


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: int
    image_key: str
    question: str
    candidates: tuple[str, ...]
    answer: str
    answer_index: int
    confidence: float


@dataclass(frozen=True)
class HistoryContext:
    text: str
    selected_turn_ids: tuple[int, ...]
    token_count: int
    stored_turn_count: int
    mode: HistoryMode
    reason: str

    @property
    def used(self) -> bool:
        return bool(self.selected_turn_ids)


def _terms(text: str) -> set[str]:
    normalized = text.casefold()
    words = {
        word for word in re.findall(r"[a-z0-9]+", normalized)
        if word not in _ENGLISH_STOPWORDS and len(word) > 1
    }
    cjk = re.findall(r"[\u3400-\u9fff]", normalized)
    words.update(character for character in cjk if character not in _CJK_STOP_CHARS)
    words.update(
        left + right
        for left, right in zip(cjk, cjk[1:])
        if left not in _CJK_STOP_CHARS or right not in _CJK_STOP_CHARS
    )
    return words


def _relevance(question: str, turn: ConversationTurn) -> float:
    query_terms = _terms(question)
    if not query_terms:
        return 0.0
    turn_terms = _terms(turn.question + " " + turn.answer)
    return len(query_terms & turn_terms) / math.sqrt(
        max(1, len(query_terms) * len(turn_terms))
    )


def _has_coreference_cue(question: str) -> bool:
    normalized = re.sub(
        r"[^a-z0-9\u3400-\u9fff]+",
        " ",
        question.casefold(),
    )
    padded = " " + normalized + " "
    return any(cue in padded for cue in _COREFERENCE_CUES)


def _history_reference_scope(question: str) -> Literal["local", "all"] | None:
    """Return the dialogue span explicitly requested by ``question``."""
    normalized = question.casefold()
    if any(re.search(pattern, normalized) for pattern in _LOCAL_HISTORY_REFERENCE_PATTERNS):
        return "local"
    if any(re.search(pattern, normalized) for pattern in _GLOBAL_HISTORY_REFERENCE_PATTERNS):
        return "all"
    return None


def _token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return max(1, math.ceil(len(text) / 3))
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return len(encoded)


class VisualConversationHistory:
    """Store many turns while retrieving a bounded same-image prompt context."""

    format_version = 1

    def __init__(
        self,
        *,
        mode: HistoryMode = "auto",
        max_stored_turns: int = 128,
        max_prompt_turns: int = 8,
        max_prompt_tokens: int = 384,
        min_relevance: float = 0.12,
    ):
        if mode not in ("off", "recent", "auto", "legacy-auto"):
            raise ValueError(
                "history mode must be off, recent, auto, or legacy-auto"
            )
        if min(max_stored_turns, max_prompt_turns, max_prompt_tokens) < 1:
            raise ValueError("history limits must be positive")
        if not 0.0 <= min_relevance <= 1.0:
            raise ValueError("min_relevance must be in [0, 1]")
        self.mode = mode
        self.max_stored_turns = max_stored_turns
        self.max_prompt_turns = min(max_prompt_turns, max_stored_turns)
        self.max_prompt_tokens = max_prompt_tokens
        self.min_relevance = min_relevance
        self.active_image: str | None = None
        self.turns: list[ConversationTurn] = []
        self._next_turn_id = 1

    def bind_image(self, image_key: str) -> bool:
        """Bind memory to one image; return True when old-image turns were reset."""
        image_key = str(Path(image_key).expanduser().resolve())
        changed = self.active_image is not None and self.active_image != image_key
        if changed:
            self.turns.clear()
            self._next_turn_id = 1
        self.active_image = image_key
        return changed

    def clear(self, *, keep_image: bool = True) -> None:
        self.turns.clear()
        self._next_turn_id = 1
        if not keep_image:
            self.active_image = None

    def add(
        self,
        *,
        question: str,
        candidates: tuple[str, ...],
        answer: str,
        answer_index: int,
        confidence: float,
    ) -> ConversationTurn:
        if self.active_image is None:
            raise ValueError("bind an image before adding a history turn")
        if not question.strip() or not answer.strip():
            raise ValueError("history question and answer must be non-empty")
        if not 0 <= answer_index < len(candidates):
            raise ValueError("history answer_index is outside candidates")
        turn = ConversationTurn(
            turn_id=self._next_turn_id,
            image_key=self.active_image,
            question=question.strip(),
            candidates=tuple(candidates),
            answer=answer.strip(),
            answer_index=answer_index,
            confidence=float(confidence),
        )
        self._next_turn_id += 1
        self.turns.append(turn)
        if len(self.turns) > self.max_stored_turns:
            self.turns = self.turns[-self.max_stored_turns :]
        return turn

    def _ranked_turns(self, question: str) -> tuple[list[ConversationTurn], str]:
        if self.mode == "off" or not self.turns:
            return [], "disabled_or_empty"
        if self.mode == "recent":
            return list(reversed(self.turns)), "recent_window"

        scored = [(_relevance(question, turn), turn) for turn in self.turns]
        if self.mode == "auto":
            history_scope = _history_reference_scope(question)
            if history_scope == "local":
                return [self.turns[-1]], "explicit_local_history_reference"
            if history_scope == "all":
                return list(reversed(self.turns)), "explicit_history_reference"

        # ``legacy-auto`` intentionally preserves the original lexical overlap
        # plus generic-coreference gate for reproducibility.
        coreference = _has_coreference_cue(question)
        relevant = [item for item in scored if item[0] >= self.min_relevance]
        if not coreference and not relevant:
            return [], "auto_skipped_unrelated_question"

        # The immediately previous turn is essential for local coreference.
        selected_ids = {self.turns[-1].turn_id} if coreference else set()
        selected_ids.update(turn.turn_id for score, turn in relevant)
        latest_turn_id = self.turns[-1].turn_id
        ranked = sorted(
            (item for item in scored if item[1].turn_id in selected_ids),
            # Local coreference must reserve budget for the immediately
            # preceding turn before lexically similar older turns.
            key=lambda item: (
                coreference and item[1].turn_id == latest_turn_id,
                item[0],
                item[1].turn_id,
            ),
            reverse=True,
        )
        reason = "coreference_and_relevance" if coreference else "relevance"
        return [turn for _, turn in ranked], reason

    @staticmethod
    def _render_turn(turn: ConversationTurn) -> str:
        return (
            f"[Turn {turn.turn_id}]\n"
            f"Question: {turn.question}\n"
            f"Selected answer: {turn.answer}"
        )

    def context(self, question: str, tokenizer: Any | None = None) -> HistoryContext:
        ranked, reason = self._ranked_turns(question)
        if not ranked:
            return HistoryContext(
                text="",
                selected_turn_ids=(),
                token_count=0,
                stored_turn_count=len(self.turns),
                mode=self.mode,
                reason=reason,
            )

        header = (
            "Conversation history for the same image. Use it only when it is "
            "needed to resolve the current question:"
        )
        selected: list[ConversationTurn] = []
        for turn in ranked:
            proposal = selected + [turn]
            chronological = sorted(proposal, key=lambda item: item.turn_id)
            text = header + "\n" + "\n".join(
                self._render_turn(item) for item in chronological
            )
            if _token_count(text, tokenizer) <= self.max_prompt_tokens:
                selected = proposal
            if len(selected) >= self.max_prompt_turns:
                break

        chronological = sorted(selected, key=lambda item: item.turn_id)
        text = (
            header + "\n" + "\n".join(self._render_turn(item) for item in chronological)
            if chronological else ""
        )
        return HistoryContext(
            text=text,
            selected_turn_ids=tuple(item.turn_id for item in chronological),
            token_count=_token_count(text, tokenizer) if text else 0,
            stored_turn_count=len(self.turns),
            mode=self.mode,
            reason=reason if chronological else "token_budget_excluded_all",
        )

    def describe(self) -> str:
        if not self.turns:
            return "Conversation history is empty."
        lines = [
            f"Conversation history: {len(self.turns)} stored turn(s) for {self.active_image}"
        ]
        lines.extend(
            f"  [{turn.turn_id}] Q: {turn.question} | A: {turn.answer} "
            f"({turn.confidence * 100:.1f}%)"
            for turn in self.turns
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "active_image": self.active_image,
            "next_turn_id": self._next_turn_id,
            "turns": [asdict(turn) for turn in self.turns],
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path

    def load(self, path: str | Path) -> None:
        path = Path(path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("format_version") != self.format_version:
            raise ValueError("unsupported conversation history format")
        turns = []
        for raw in payload.get("turns", []):
            raw = dict(raw)
            raw["candidates"] = tuple(raw["candidates"])
            turns.append(ConversationTurn(**raw))
        self.active_image = payload.get("active_image")
        self.turns = turns[-self.max_stored_turns :]
        self._next_turn_id = max(
            int(payload.get("next_turn_id", 1)),
            max((turn.turn_id for turn in self.turns), default=0) + 1,
        )


def format_candidate_with_history(
    question: str,
    candidate: str,
    context: HistoryContext,
    *,
    prompt_policy: HistoryPromptPolicy = "legacy-image-only",
) -> str:
    if prompt_policy not in ("adaptive", "legacy-image-only"):
        raise ValueError("unknown history prompt policy")
    prefix = context.text + "\n\nCurrent turn:\n" if context.text else ""
    if context.text and prompt_policy == "adaptive":
        instruction = (
            "Judge whether the candidate is supported by the conversation "
            "history; use image evidence only when it is relevant."
        )
    else:
        instruction = "Judge whether the candidate is supported by the image."
    return (
        f"{prefix}Question: {question}\n"
        f"Candidate: {candidate}\n"
        f"{instruction}"
    )
