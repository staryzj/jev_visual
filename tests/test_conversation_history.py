from jev.conversation_history import (
    VisualConversationHistory,
    format_candidate_with_history,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)


def add_turn(history, question, answer):
    candidates = (answer, "other")
    return history.add(
        question=question,
        candidates=candidates,
        answer=answer,
        answer_index=0,
        confidence=0.8,
    )


def test_auto_history_uses_previous_turn_for_coreference(tmp_path):
    history = VisualConversationHistory(mode="auto")
    history.bind_image(tmp_path / "dog.png")
    add_turn(history, "这是什么动物？", "狗")

    context = history.context("它是什么品种？")

    assert context.used is True
    assert context.selected_turn_ids == (1,)
    assert "Selected answer: 狗" in context.text
    assert context.reason == "coreference_and_relevance"


def test_auto_history_skips_an_unrelated_question(tmp_path):
    history = VisualConversationHistory(mode="auto", min_relevance=0.5)
    history.bind_image(tmp_path / "dog.png")
    add_turn(history, "这是什么动物？", "狗")

    context = history.context("背景是什么颜色？")

    assert context.used is False
    assert context.reason == "auto_skipped_unrelated_question"


def test_auto_history_recognizes_explicit_previous_question_reference(tmp_path):
    history = VisualConversationHistory(mode="auto")
    history.bind_image(tmp_path / "dog.png")
    add_turn(history, "这是什么品种？", "柴犬")

    context = history.context("上一个问题是什么？")

    assert context.used is True
    assert context.selected_turn_ids == (1,)
    assert context.reason == "explicit_local_history_reference"
    assert "Question: 这是什么品种？" in context.text


def test_auto_history_recognizes_multi_turn_reference(tmp_path):
    history = VisualConversationHistory(mode="auto")
    history.bind_image(tmp_path / "dog.png")
    add_turn(history, "第一轮问了颜色吗？", "没有")
    add_turn(history, "第二轮问了品种吗？", "是")

    context = history.context("第二轮里我问了什么？")

    assert context.used is True
    assert context.selected_turn_ids == (1, 2)
    assert context.reason == "explicit_history_reference"


def test_legacy_auto_preserves_old_history_gate(tmp_path):
    history = VisualConversationHistory(mode="legacy-auto")
    history.bind_image(tmp_path / "dog.png")
    add_turn(history, "这是什么品种？", "柴犬")

    context = history.context("上一个问题是什么？")

    assert context.used is False
    assert context.reason == "auto_skipped_unrelated_question"


def test_history_resets_when_image_changes(tmp_path):
    history = VisualConversationHistory(mode="recent")
    history.bind_image(tmp_path / "one.png")
    add_turn(history, "Question one", "answer one")

    changed = history.bind_image(tmp_path / "two.png")

    assert changed is True
    assert history.turns == []


def test_recent_history_obeys_turn_and_token_budgets(tmp_path):
    history = VisualConversationHistory(
        mode="recent", max_prompt_turns=2, max_prompt_tokens=180
    )
    history.bind_image(tmp_path / "dog.png")
    for index in range(5):
        add_turn(history, f"Question {index}", f"answer {index}")

    context = history.context("follow up", CharacterTokenizer())

    assert len(context.selected_turn_ids) <= 2
    assert context.token_count <= 180
    assert context.selected_turn_ids == tuple(sorted(context.selected_turn_ids))


def test_history_round_trip_and_candidate_prompt(tmp_path):
    source = VisualConversationHistory(mode="auto")
    source.bind_image(tmp_path / "dog.png")
    add_turn(source, "What animal is this?", "dog")
    path = source.save(tmp_path / "history.json")

    restored = VisualConversationHistory(mode="auto")
    restored.load(path)
    context = restored.context("What breed is it?")
    prompt = format_candidate_with_history("What breed is it?", "shiba", context)

    assert restored.to_dict() == source.to_dict()
    assert "What animal is this?" in prompt
    assert "Selected answer: dog" in prompt
    assert prompt.endswith(
        "Judge whether the candidate is supported by the image."
    )

    adaptive_prompt = format_candidate_with_history(
        "What breed is it?", "shiba", context, prompt_policy="adaptive"
    )
    assert adaptive_prompt.endswith(
        "Judge whether the candidate is supported by the conversation history; "
        "use image evidence only when it is relevant."
    )
