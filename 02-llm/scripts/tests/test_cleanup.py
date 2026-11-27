"""Fixed reply cleanup used for every rater-facing reply (analysis plan v1.8, v1.9)."""

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


def test_strips_markdown_emphasis_but_keeps_underscores_and_lone_stars():
    out = clean_reply("2024 was a *big* year, and **this** is __key__.")
    assert out.text == "2024 was a big year, and this is key." and out.ops == ("markdown",)
    kept = "Ask u/some_name_here, or rate it 5 * 2."
    assert clean_reply(kept).text == kept and clean_reply(kept).ops == ()


def test_markdown_around_opener_is_removed_before_the_opener():
    out = clean_reply("*Uy*, gets ko yan.")
    assert out.text == "Gets ko yan." and out.ops == ("markdown", "opener")


def test_dashes_become_commas_or_hyphens():
    out = clean_reply("Mahirap talaga \u2013 lalo na pag pamilya \u2014 pero kaya mo. Mga 2\u20133 buwan lang.")
    assert out.text == "Mahirap talaga, lalo na pag pamilya, pero kaya mo. Mga 2-3 buwan lang."
    assert out.ops == ("dash",)


def test_community_style_reply_without_tells_is_unchanged():
    text = "Kausapin mo muna siya ng maayos. If ayaw niya, then leave - you deserve better."
    assert clean_reply(text).text == text and clean_reply(text).ops == ()


def test_straightens_quotes_and_collapses_spacing():
    out = clean_reply("Kaya mo \u2018yan.  Talaga.   \n\n\n\u201cOk\u201d lang.")
    assert out.text == "Kaya mo 'yan. Talaga.\n\n\"Ok\" lang." and out.ops == ("quotes", "spacing")


def test_removes_sentence_initial_honestly_only():
    out = clean_reply("Honestly, ang hirap. Pero honestly, kaya mo. I honestly think so.")
    assert out.text == "Ang hirap. Pero honestly, kaya mo. I honestly think so." and out.ops == ("honestly",)
    assert clean_reply("Mahirap. Honestly, it is.").text == "Mahirap. It is."


def test_drops_stock_signoff_sentences_but_keeps_content():
    out = clean_reply("Mag-ipon ka muna. Good luck with your shopping! Kaya mo 'yan, OP!")
    assert out.text == "Mag-ipon ka muna." and out.ops == ("signoff",)
    kept = "Mag-ipon ka muna. Good luck with the shopping, I hope he loves it!"
    assert clean_reply(kept).text == kept
    assert clean_reply("Good luck!").text == "Good luck!"  # never empties a reply
