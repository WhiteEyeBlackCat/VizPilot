"""Review round 3: the wording gate is a token allowlist (fail closed by
construction). Fresh adversarial strings that a denylist missed, the correct
set the gate must keep, and a property test: random sentences built from a
broad English vocabulary are accepted only when every word is allowlisted, and
no accepted sentence states the wrong group order.
"""

import random
import re
from types import SimpleNamespace

import polars as pl
import pytest

from app.llm import claims as C
from app.llm.claims import build_claim, semantic_problem
from tests.test_semantic_claims import _rel_run, _relationship_frame, _wording
from tests.test_wording_attacks import _frame, _run, _time_source, time_data  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def data():
    from app.profiling.profiler import profile_dataset

    df = _frame()
    return df, profile_dataset(df, "7" * 32, 10**6)


FRESH_GROUP = [
    "North's revenue is the envy of the others.", "South trails behind.", "North takes the crown.",
    "Revenue is concentrated in North.", "Most revenue comes from North.", "North accounts for the bulk of revenue.",
    "North is ahead.", "Behind North comes South.", "Ranking: North, North East, West, South.",
    "North is way ahead of the pack.", "North is in front on revenue.", "South brings up the rear.",
    "North wins on revenue.", "North reigns supreme in revenue.", "North dwarfs South.", "North towers over South.",
    "North overshadows South.", "South pales next to North.", "North is number one for revenue.",
    "North averages about two hundred.", "North's mean revenue is roughly two hundred.", "North sells 50% more than South.",
    "South has the highest revenue. North is ahead of West.", "South has the highest revenue. North is runner-up.",
    "South (the lowest) and North (the highest).", "South has the highest revenue — no, North does.",
    "North Is Ahead Of The Pack", "Revenue Peaks In The North", "North Wins The Revenue Race", "\"North\" is ahead.",
    "Norths sell more than Souths.", "N. region sells more than S. region.", "North 🚀 South 📉",
    "North vende mas que South.", "North verkauft mehr als South.", "North vend plus que South.",
    "North is mayor than South.", "North beats out the rest.", "North is unrivalled in revenue.",
    "South is the weakest link in revenue.", "Revenue in North is unmatched.", "South underwhelms relative to North.",
    "North is the clear frontrunner.", "South finishes behind North.", "Region is behind the revenue gap.",
    "Region underlies revenue.", "Revenue varies on account of region.", "Region spurs revenue.",
    "Region fuels revenue in South.", "Region dampens revenue in North.", "Region governs revenue.",
    "Region dictates revenue.", "Region gives rise to revenue differences.", "Revenue hinges on region.",
    "Region powers revenue.", "Region translates into revenue.",
]
FRESH_FIELDS = [("headline", "North Is Ahead"), ("headline", "North Takes The Crown"),
                ("headline", "Revenue Concentrated Up North"), ("why", "North is ahead, so invest there."),
                ("why", "Region fuels revenue.")]


@pytest.mark.parametrize("text", FRESH_GROUP)
def test_fresh_group_attacks(data, text) -> None:
    _, w, ins = _run(data, text=text)
    assert ins is not None and ins["text"] != text


@pytest.mark.parametrize("field, text", FRESH_FIELDS)
def test_fresh_headline_and_why_attacks(data, field, text) -> None:
    kw = {"headline": text} if field == "headline" else {"why": text}
    _, _, ins = _run(data, **kw)
    assert ins is not None and ins.get("headline" if field == "headline" else "why_it_matters") != text


@pytest.fixture(scope="module")
def pos():
    return _relationship_frame({"north": 2.0, "south": 1.5, "east": 2.5, "west": 1.8})


@pytest.fixture(scope="module")
def flip():
    return _relationship_frame({"north": 2.0, "south": 2.0, "east": -2.0, "west": -2.0})


def _shipped(frame, text) -> bool:
    result, _ = _rel_run(frame, text)
    return _wording(result)["source"] == "llm2"


@pytest.mark.parametrize("text", ["w eases as z climbs.", "w softens as z grows.", "w retreats as z advances.",
                                  "w and z diverge.", "w runs counter to z.", "There is a trade-off between z and w.",
                                  "w and z pull in different directions.", "w is a mirror image of z."])
def test_fresh_positive_relationship_attacks(pos, text) -> None:
    assert not _shipped(pos, text)


@pytest.mark.parametrize("text", ["w rises with z throughout the regions.", "w rises with z region by region.",
                                  "w and z are tied together in each and every region.", "w climbs with z wherever you look.",
                                  "w and z are positively linked overall.", "w and z rise together.", "Higher z pairs with higher w."])
def test_fresh_sign_flip_attacks(flip, text) -> None:
    assert not _shipped(flip, text)


@pytest.mark.parametrize("text", ["load ballooned after the change.", "load swelled later.", "load escalated after the change.",
                                  "load mounted after the change.", "load ramped up after the change.", "load took off later.",
                                  "load hit new highs later.", "Load Expanded After The Change", "load was elevated after the change.",
                                  "load is heavier now."])
def test_fresh_time_attacks(time_data, text) -> None:  # noqa: F811
    assert _time_source(time_data, 1, text) != "llm2"


@pytest.mark.parametrize("text", ["u crests in the middle of x.", "u tops out at middle x.", "u forms an arch over x.",
                                  "u forms a mound over x.", "u is highest at mid-range x.", "u heads downhill as x increases.",
                                  "u is lowest at the edges of x."])
def test_fresh_curve_attacks(time_data, text) -> None:  # noqa: F811
    assert _time_source(time_data, 2, text) != "llm2"


@pytest.fixture(scope="module")
def means(data):
    df, _ = data
    return {r: f"{m:.1f}" for r, m in df.group_by("region").agg(pl.col("revenue").mean()).iter_rows()}


KEPT_GROUP = [
    "South leads all other regions on revenue.", "Revenue is highest in South and lowest in North.",
    "South generates the most revenue, North the least.", "Mean revenue ranges from {North} in North to {South} in South.",
    "North has the lowest average revenue ({North}).", "South averages {South}, about twice North's {North}.",
    "South's average revenue is well above the other regions.", "There is a large revenue gap between South and North.",
    "Regional differences in revenue are substantial.", "West and North trail South and North East.",
    "North East ranks second behind South.", "Revenue differs markedly by region, with South at the top.",
    "Revenue varies by region; South stands out.", "South outperforms North by a wide margin.",
    "North lags the other regions on revenue.",
]


@pytest.mark.parametrize("text", KEPT_GROUP)
def test_correct_group_wording_is_kept(data, means, text) -> None:
    text = text.format(**means)
    result, _, ins = _run(data, text=text)
    assert ins is not None and ins["text"] == text, [e for e in result["debug"]["errors"] if "insight 1" in e]


@pytest.mark.parametrize("text", ["South Leads Regional Revenue", "Revenue Varies Sharply By Region"])
def test_correct_headlines_are_kept(data, text) -> None:
    _, _, ins = _run(data, headline=text)
    assert ins is not None and ins["headline"] == text


def test_rejected_why_costs_only_the_why(data) -> None:
    result, _, ins = _run(data, text="South has the highest revenue.", why="Knowing which regions earn more helps target investment.")
    assert ins["text"] == "South has the highest revenue." and ins["why_it_matters"] != "Knowing which regions earn more helps target investment."
    assert any(e.startswith("llm2 why 1") for e in result["debug"]["errors"])


@pytest.mark.parametrize("frame, text", [
    ("pos", "w rises with z in every region."), ("pos", "Within each region, higher z goes with higher w."),
    ("pos", "The z-w relationship is positive in all four regions."),
    ("flip", "w rises with z in north and south but falls in east and west."),
    ("flip", "The sign of the z-w relationship depends on the region."),
])
def test_correct_relationship_wording_is_kept(request, frame, text) -> None:
    assert _shipped(request.getfixturevalue(frame), text)


def test_depends_on_is_causal_outside_sign_findings(pos) -> None:
    assert not _shipped(pos, "The sign of the z-w relationship depends on the region.")  # positive claim: no sign varies
    assert not _shipped(pos, "w depends on z.")


@pytest.mark.parametrize("text", ["load fell sharply around 2023-11-01.", "load was lower in the later period."])
def test_correct_time_wording_is_kept(time_data, text) -> None:  # noqa: F811
    assert _time_source(time_data, 1, text) == "llm2"


@pytest.mark.parametrize("text", ["u is U-shaped in x.", "u has a U-shaped relationship with x."])
def test_correct_curve_wording_is_kept(time_data, text) -> None:  # noqa: F811
    assert _time_source(time_data, 2, text) == "llm2"


# --- property: nothing outside the allowlist is ever accepted ------------------------------

_VOCAB = """
accept achieve add admire advance advise affect agree allow announce appear apply argue arrange arrive ask attack
avoid bake balance ban bang bat battle beam beg behave belong bleach bless blind blink blot blush boast boil bolt
bomb book bore borrow bounce bow box brake branch breathe bruise brush bubble bump burn bury buzz calculate call
camp care carry carve cause challenge change charge chase cheat check cheer chew choke chop claim clap clean clear
climb clip close coach coil collect colour comb command communicate compare compete complain complete concentrate
concern confess confuse connect consider consist contain continue copy correct cough count cover crack crash crawl
cross crush cry cure curl curve cycle dam damage dance dare decay deceive decide decorate delay delight deliver
depend describe desert deserve destroy detect develop disagree disappear disapprove disarm discover dislike divide
double doubt drag drain dream dress drip drop drown drum dry dust earn educate embarrass employ empty encourage
end enjoy enter entertain escape examine excite excuse exercise exist expand expect explain explode extend face
fade fail fancy fasten fax fear fence fetch file fill film fire fit fix flap flash float flood flow flower fold
follow fool force form found frame frighten fry gather gaze glow glue grab grate grease greet grin grip groan
guarantee guard guess guide hammer hand handle hang happen harm hate haunt head heal heap heat help hook hop hope
hover hug hum hunt hurry identify ignore imagine impress improve include increase influence inform inject injure
instruct intend interest interfere interrupt introduce invent invite irritate itch jail jam jog join joke judge
juggle jump kick kill kiss kneel knit knock knot label land last laugh launch learn level license lick lie
lighten like list listen live load lock long look love manage march mark marry match mate matter measure melt
memorise mend milk mine miss mix moan moor mourn move muddle mug multiply murder nail name need nest nod note
notice number obey object observe obtain occur offend offer open order overflow owe own pack paddle paint park
part pass paste pat pause peck pedal peel peep perform permit phone pick pinch pine place plan plant play please
plug point poke polish pop possess post pour practise pray preach precede prefer prepare present preserve press
pretend prevent prick print produce program promise protect provide pull pump punch puncture punish push
question queue race radiate rain raise reach realise receive recognise record reduce reflect refuse regret reign
reject rejoice relax release rely remain remember remind remove repair repeat replace reply report reproduce
request rescue retire return rhyme rinse risk rob rock roll rot rub ruin rule rush sack sail satisfy save saw
scare scatter scold scorch scrape scratch scream screw scribble scrub seal search separate serve settle shade
share shave shelter shiver shock shop shrug sigh sign signal sin sip ski skip slap slip slow smash smell smile
smoke snatch sneeze sniff snore snow soak soothe sound spare spark sparkle spell spill spoil spot spray sprout
squash squeak squeal squeeze stain stamp stare start stay steer step stir stitch stop store strap strengthen
stretch strip stroke stuff subtract succeed suck suffer suggest suit supply support suppose surprise surround
suspect suspend switch talk tame tap taste tease telephone tempt terrify test thank thaw tick tickle tie time
tip tire touch tour tow trace trade train transport trap travel treat tremble trick trip trot trouble trust try
tug tumble turn twist type undress unfasten unite unlock unpack untidy use vanish visit wail wait walk wander want
warm warn wash waste watch water wave weigh welcome whine whip whirl whisper whistle wink wipe wish wobble wonder
work worry wrap wreck wrestle wriggle yawn yell zip zoom big small huge tiny great awful bright dull eager famous
gentle happy jolly kind lively nice proud silly witty brave calm fierce bold dominant supreme stellar leading
ahead behind atop beneath crown king queen champion winner loser bulk lion share mostly mainly chiefly largely
""".split()


def _region_claim(df):
    rows = [{"group": r, "n": int(n), "mean": float(m)} for r, n, m in
            df.group_by("region").agg(pl.len(), pl.col("revenue").mean()).iter_rows()]
    return build_claim(SimpleNamespace(test_type="group_difference", columns={"group": "region", "target": "revenue"},
                                       evidence={"groups": rows}))


def test_property_only_allowlisted_and_correct_sentences_are_accepted(data) -> None:
    df, _ = data
    claim = _region_claim(df)
    means = {g.label: g.mean for g in claim.groups}
    labels = list(means)
    names = {"region", "revenue", "units", "channel"}
    allowed = C._FUNCTION_WORDS | C._NEUTRAL_WORDS | C._ADVISORY_WORDS | C._LEXICON_WORDS | C._claim_words(claim)
    lexicon = sorted(C._COMP_UP | C._COMP_DOWN | C._SUP_UP | C._SUP_DOWN | {"has", "the", "than", "in", "is", "revenue"})
    rng = random.Random(17)
    shapes = [
        "{a} {w1} {w2} revenue than {b}.",
        "{a} {w1} the {w2} revenue.",
        "revenue {w1} {w2} in {a}.",
        "{a} {w1} {b} on revenue.",
        "{a} is {w1} {w2} {b}.",
    ]
    accepted = 0
    for _ in range(1500):
        a, b = rng.sample(labels, 2)
        pool = _VOCAB if rng.random() < 0.6 else lexicon
        sentence = rng.choice(shapes).format(a=a, b=b, w1=rng.choice(pool), w2=rng.choice(_VOCAB + lexicon))
        if semantic_problem(sentence, claim, names) is not None:
            continue
        accepted += 1
        stripped = sentence.lower()
        for lab in sorted(labels, key=len, reverse=True):
            stripped = re.sub(rf"\b{re.escape(lab.lower())}\b", " ", stripped)
        for word in re.findall(r"[a-z]+", stripped):
            assert word in allowed or word in names, (sentence, word)
        # independent oracle: "<a> <comparative> ... than <b>" must match the means
        words = set(re.findall(r"[a-z]+", stripped))
        if " than " in sentence and words & (C._COMP_UP | C._COMP_DOWN):
            up = bool(words & C._COMP_UP)
            assert (means[a] > means[b]) == up, sentence
        if words & (C._SUP_UP | C._SUP_DOWN) and sentence.startswith(a) and " than " not in sentence:
            top = max(means, key=means.get) if words & C._SUP_UP else min(means, key=means.get)
            assert a == top, sentence
    assert accepted >= 20  # the property is exercised on real accepted sentences
