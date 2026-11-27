"""Fixed reply cleanup used for every rater-facing reply (analysis plan v1.8)."""

from payo_llm.cleanup import clean_reply


def test_strips_stacked_interjections_and_emoji_then_capitalizes():
    out = clean_reply("Uy, naku, gets ko yung frustration mo! You're beautiful as you are! 😊")
    assert out.text == "Gets ko yung frustration mo! You're beautiful as you are!"
    assert out.ops == ("emoji", "opener")


def test_okay_so_opener_and_content_words_are_kept():
    assert clean_reply("Okay, so this is tricky.").text == "This is tricky."
    assert clean_reply("Okay so this is tricky.").text == "This is tricky."
    kept = clean_reply("Grabe yung pinagdadaanan mo.")
    assert kept.text == "Grabe yung pinagdadaanan mo." and kept.ops == ()


def test_trims_to_last_whole_sentence_within_cap():
    text = " ".join(["Ito ay pangungusap na may anim."] * 30)  # 6 words per sentence, 180 words
    out = clean_reply(text, max_words=110)
    assert out.words == 108 and out.text.endswith(".") and out.ops == ("trimmed",)


def test_drops_unfinished_sentence_only_when_truncated():
    text = "Kaya mo yan. Talk to him and"
    assert clean_reply(text, truncated=True).text == "Kaya mo yan."
    assert clean_reply(text, truncated=False).text == text


def test_hard_cut_when_one_sentence_exceeds_cap():
    out = clean_reply(" ".join(["salita"] * 150), max_words=110)
    assert out.words == 110 and out.ops == ("hard_cut",)


def test_short_clean_reply_is_unchanged():
    text = "Finish your degree para may fallback ka in case shit happens"
    assert clean_reply(text).text == text and clean_reply(text).ops == ()
